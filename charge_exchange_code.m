function MCC_App()
    % MCC_APP: Monte Carlo Collision (MCC) Simulator for Charge Exchange (CEX)
    % Features: Directional Probes, Log Scales, CSV Export, and Live GIF Recording.

    %% 1. Initialize Global Simulation State
    sim.isRunning = false;
    sim.dt = 1e-7;
    sim.m_XE = 2.18e-25;           % 131.3 amu, Xenon mass
    sim.q_m = 1.602e-19 / sim.m_XE; 
    sim.neut_vth = sqrt(2 * 1.38e-23 / sim.m_XE);
    
    % Particle arrays 
    p_x = []; p_y = []; p_u = []; p_v = []; p_cex = false(0);

    % Detector Arrays
    detectors_Z = []; detectors_R = []; detectors_rad = []; detectors_ang = []; 
    det_step_counts = []; 
    
    h_det_markers = gobjects(0); h_det_normals = gobjects(0); 
    h_det_texts = gobjects(0); h_det_lines = gobjects(0);
    
    time_history = []; det_counts_history = {}; color_palette = lines(10); 
    
    % GIF Recording State
    recorded_frames = {}; 

    % Grid definitions for Density maps
    num_bins_Z = 60; num_bins_R = 40;
    Z_edges = linspace(-0.5, 2.0, num_bins_Z + 1); R_edges = linspace(-1.0, 1.0, num_bins_R + 1);
    Z_centers = Z_edges(1:end-1) + diff(Z_edges)/2; R_centers = R_edges(1:end-1) + diff(R_edges)/2;

    %% 2. Setup GUI Layout 
    fig = figure('Name', 'MCC Charge Exchange Simulator', ...
                 'NumberTitle', 'off', 'Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.8], ...
                 'CloseRequestFcn', @onClose, 'Color', 'w');
             
    % Left Panel: Controls 
    uicontrol('Style', 'text', 'Units', 'normalized', 'Position', [0.02, 0.83, 0.16, 0.03], 'String', 'Thruster Diameter (m):', 'HorizontalAlignment', 'left', 'FontWeight', 'bold', 'BackgroundColor', 'w');
    edit_diam = uicontrol('Style', 'edit', 'Units', 'normalized', 'Position', [0.19, 0.83, 0.08, 0.04], 'String', '0.3');
    
    uicontrol('Style', 'text', 'Units', 'normalized', 'Position', [0.02, 0.77, 0.16, 0.03], 'String', 'Neutral Density n0 (m^-3):', 'HorizontalAlignment', 'left', 'FontWeight', 'bold', 'BackgroundColor', 'w');
    edit_n0 = uicontrol('Style', 'edit', 'Units', 'normalized', 'Position', [0.19, 0.77, 0.08, 0.04], 'String', '1e18');
    
    btn_toggle = uicontrol('Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.02, 0.68, 0.25, 0.06], ...
                           'String', 'START SIMULATION', 'FontWeight', 'bold', 'BackgroundColor', [0.8 1 0.8], 'Callback', @toggleSim);
                       
    uicontrol('Style', 'text', 'Units', 'normalized', 'Position', [0.02, 0.61, 0.25, 0.03], 'String', '--- ANALYSIS TARGET ---', 'FontWeight', 'bold', 'BackgroundColor', 'w');
    popup_target = uicontrol('Style', 'popupmenu', 'Units', 'normalized', 'Position', [0.02, 0.57, 0.25, 0.04], 'String', {'CEX Ions (Red)', 'Primary Ions (Blue)'});
                             
    % Detector Controls
    uicontrol('Style', 'text', 'Units', 'normalized', 'Position', [0.02, 0.51, 0.25, 0.03], 'String', '--- DIRECTIONAL PROBES ---', 'FontWeight', 'bold', 'BackgroundColor', 'w');
    
    uicontrol('Style', 'text', 'Units', 'normalized', 'Position', [0.02, 0.47, 0.16, 0.03], 'String', 'Probe Radius (m):', 'HorizontalAlignment', 'left', 'BackgroundColor', 'w');
    edit_detRad = uicontrol('Style', 'edit', 'Units', 'normalized', 'Position', [0.19, 0.47, 0.08, 0.04], 'String', '0.05');

    uicontrol('Style', 'text', 'Units', 'normalized', 'Position', [0.02, 0.43, 0.16, 0.03], 'String', 'Probe Angle (deg):', 'HorizontalAlignment', 'left', 'BackgroundColor', 'w');
    edit_detAng = uicontrol('Style', 'edit', 'Units', 'normalized', 'Position', [0.19, 0.43, 0.08, 0.04], 'String', '0');

    uicontrol('Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.02, 0.38, 0.12, 0.04], 'String', '+ Add Probe', 'BackgroundColor', [0.8 0.9 1], 'Callback', @addDetector);
    uicontrol('Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.15, 0.38, 0.12, 0.04], 'String', 'Clear All', 'BackgroundColor', [1 0.8 0.8], 'Callback', @clearDetectors);
              
    % Export Tools
    uicontrol('Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.02, 0.31, 0.25, 0.04], 'String', 'Export Data to CSV', 'FontWeight', 'bold', 'BackgroundColor', [1 0.9 0.6], 'Callback', @exportData);
    
    % GIF Recording Tools
    chk_recordGif = uicontrol('Style', 'checkbox', 'Units', 'normalized', 'Position', [0.02, 0.26, 0.25, 0.03], 'String', 'Record GIF (Frames: 0)', 'FontWeight', 'bold', 'BackgroundColor', 'w', 'ForegroundColor', [0.8 0 0]);
    uicontrol('Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.02, 0.22, 0.25, 0.04], 'String', 'Save GIF Animation', 'FontWeight', 'bold', 'BackgroundColor', [0.9 0.8 1], 'Callback', @saveGif);

    % Plot Scales Toggles
    uicontrol('Style', 'text', 'Units', 'normalized', 'Position', [0.02, 0.16, 0.25, 0.03], 'String', '--- TIME PLOT SCALES ---', 'FontWeight', 'bold', 'BackgroundColor', 'w');
    chk_logX = uicontrol('Style', 'checkbox', 'Units', 'normalized', 'Position', [0.05, 0.12, 0.1, 0.04], 'String', 'Log X Axis', 'BackgroundColor', 'w', 'Callback', @updateScales);
    chk_logY = uicontrol('Style', 'checkbox', 'Units', 'normalized', 'Position', [0.16, 0.12, 0.1, 0.04], 'String', 'Log Y Axis', 'BackgroundColor', 'w', 'Callback', @updateScales);

    txt_status = uicontrol('Style', 'text', 'Units', 'normalized', 'Position', [0.02, 0.05, 0.25, 0.05], ...
                           'String', 'Active Particles: 0 | Iteration: 0', 'HorizontalAlignment', 'left', 'FontSize', 10, 'BackgroundColor', 'w');

    % Right Panel: Axes Stacked (4 Plots)
    ax_scatter = axes('Parent', fig, 'Position', [0.35, 0.78, 0.53, 0.17]);
    hold(ax_scatter, 'on'); grid(ax_scatter, 'on'); box(ax_scatter, 'on'); axis(ax_scatter, [-0.5, 2, -1, 1]);
    ylabel(ax_scatter, 'r (m)'); title(ax_scatter, 'Live Particle Flow (Blue: Primary, Red: CEX)');
    h_beam = scatter(ax_scatter, [], [], 2, 'b', 'filled', 'MarkerFaceAlpha', 0.5); h_cex  = scatter(ax_scatter, [], [], 4, 'r', 'filled'); 

    ax_color = axes('Parent', fig, 'Position', [0.35, 0.54, 0.53, 0.17]);
    hold(ax_color, 'on'); box(ax_color, 'on'); axis(ax_color, [-0.5, 2, -1, 1]);
    ylabel(ax_color, 'r (m)'); title(ax_color, '2D Density Map');
    h_map = imagesc(ax_color, Z_centers, R_centers, zeros(num_bins_R, num_bins_Z));
    set(ax_color, 'YDir', 'normal'); colormap(ax_color, 'hot'); 
    cb = colorbar(ax_color, 'Position', [0.90, 0.54, 0.015, 0.17]); ylabel(cb, 'Relative Count');

    ax_1D = axes('Parent', fig, 'Position', [0.35, 0.30, 0.53, 0.17]);
    hold(ax_1D, 'on'); grid(ax_1D, 'on'); box(ax_1D, 'on'); xlim(ax_1D, [-0.5, 2]);
    ylabel(ax_1D, '1D Density'); title(ax_1D, 'Axial Density Profile');
    h_1D = plot(ax_1D, Z_centers, zeros(size(Z_centers)), 'r-', 'LineWidth', 2);

    ax_time = axes('Parent', fig, 'Position', [0.35, 0.06, 0.53, 0.17]);
    hold(ax_time, 'on'); grid(ax_time, 'on'); box(ax_time, 'on'); xlim(ax_time, [0, 1000]);
    xlabel(ax_time, 'Iteration'); ylabel(ax_time, 'Particle Flux'); title(ax_time, 'Probe Measurements Over Time'); 

    %% 3. Main Simulation Loop
    iteration = 0;
    while ishandle(fig)
        if sim.isRunning
            iteration = iteration + 1;
            
            rT = str2double(get(edit_diam, 'String')) / 2;
            n0 = str2double(get(edit_n0, 'String'));
            
            % --- INJECT PARTICLES ---
            num_inject = 25;
            p_x = [p_x; zeros(num_inject, 1)];
            p_y = [p_y; rT * (-1 + 2 * rand(num_inject, 1))];
            p_u = [p_u; repmat(29000, num_inject, 1)];
            p_v = [p_v; -2000 + 4000 * rand(num_inject, 1)];
            p_cex = [p_cex; false(num_inject, 1)];
            
            % --- ACCELERATE ---
            E0 = 50; theta = 5 * (pi/180); Ex = -sin(theta) * E0; Ey = cos(theta) * E0;
            accel_mask = (p_x < 4*rT);
            p_u(accel_mask) = p_u(accel_mask) + sim.q_m * Ex * sim.dt;
            sign_y = ones(size(p_y)); sign_y(p_y < 0) = -1;
            p_v(accel_mask) = p_v(accel_mask) + sim.q_m * (Ey .* sign_y(accel_mask)) * sim.dt;
            
            % --- MOVE ---
            p_x = p_x + p_u * sim.dt; p_y = p_y + p_v * sim.dt;
            
            % --- MCC COLLISION ---
            a_val = 1 / (1 - 1/sqrt(2));
            R_dist = sqrt(p_y.^2 + (p_x + rT).^2); theta_ang = atan2(abs(p_y), (p_x + rT));
            nn = n0 .* a_val .* (1 - 1./sqrt(1 + (rT./R_dist).^2)) .* cos(theta_ang);
            g = sqrt(p_u.^2 + p_v.^2); a_cs = -0.8821 * log(g) + 15.1262; sigma = (a_cs.^2) * 1e-20;
            P = 1 - exp(-nn .* sigma .* g .* sim.dt); collide_mask = rand(length(p_x), 1) < P;
            
            n_coll = sum(collide_mask);
            if n_coll > 0
                fmaxw_u = 2 .* (sum(rand(n_coll, 3), 2) - 1.5); fmaxw_v = 2 .* (sum(rand(n_coll, 3), 2) - 1.5);
                p_u(collide_mask) = sim.neut_vth .* fmaxw_u; p_v(collide_mask) = sim.neut_vth .* fmaxw_v;
                p_cex(collide_mask) = true;
            end
            
            % --- DETECTOR INTERCEPTION ---
            target_idx = get(popup_target, 'Value');
            for d = 1:length(detectors_Z)
                if isempty(p_x), break; end 
                dx = p_x - detectors_Z(d); dy = p_y - detectors_R(d);
                ang = detectors_ang(d); rad = detectors_rad(d);
                
                dn = -dx .* cos(ang) + dy .* sin(ang); 
                dt_val = dx .* sin(ang) + dy .* cos(ang); 
                
                hit_mask = abs(dn) <= 0.005 & abs(dt_val) <= rad;
                
                if any(hit_mask)
                    if target_idx == 1
                        det_step_counts(d) = det_step_counts(d) + sum(hit_mask & p_cex);
                    else
                        det_step_counts(d) = det_step_counts(d) + sum(hit_mask & ~p_cex);
                    end
                    p_x(hit_mask) = []; p_y(hit_mask) = [];
                    p_u(hit_mask) = []; p_v(hit_mask) = []; p_cex(hit_mask) = [];
                end
            end
            
            % --- FILTER BOUNDS ---
            in_domain = (p_x >= -0.5) & (p_x <= 2) & (p_y >= -1) & (p_y <= 1);
            p_x = p_x(in_domain); p_y = p_y(in_domain);
            p_u = p_u(in_domain); p_v = p_v(in_domain); p_cex = p_cex(in_domain);
            
            % --- UPDATE GUI (Every 25 iterations) ---
            if mod(iteration, 25) == 0
                set(h_beam, 'XData', p_x(~p_cex), 'YData', p_y(~p_cex));
                set(h_cex, 'XData', p_x(p_cex), 'YData', p_y(p_cex));
                
                if target_idx == 1
                    active_mask = p_cex; cmap_choice = 'hot'; plot_color = 'r-'; title_prefix = 'CEX';
                else
                    active_mask = ~p_cex; cmap_choice = 'parula'; plot_color = 'b-'; title_prefix = 'Primary Ion';
                end

                if any(active_mask)
                    [N_map, ~, ~] = histcounts2(p_x(active_mask), p_y(active_mask), Z_edges, R_edges);
                    set(h_map, 'CData', N_map'); colormap(ax_color, cmap_choice);
                    title(ax_color, sprintf('2D %s Density Map', title_prefix));
                    
                    N_1D = sum(N_map, 2); set(h_1D, 'YData', N_1D, 'Color', plot_color(1)); 
                    title(ax_1D, sprintf('Axial %s Density Profile', title_prefix));
                    if max(N_1D) > 0, ylim(ax_1D, [0, max(N_1D) * 1.1]); end
                else
                    set(h_map, 'CData', zeros(num_bins_R, num_bins_Z)); set(h_1D, 'YData', zeros(size(Z_centers)));
                end
                
                % Update Time Series Plot
                time_history(end+1) = iteration;
                for i = 1:length(detectors_Z)
                    flux_val = det_step_counts(i); set(h_det_texts(i), 'String', sprintf('%d', flux_val));
                    det_counts_history{i}(end+1) = flux_val;
                    set(h_det_lines(i), 'XData', time_history, 'YData', det_counts_history{i});
                end
                if ~isempty(det_step_counts), det_step_counts(:) = 0; end
                
                % Safe X-Axis Auto-Scrolling (Prevents Log-0 crashes)
                if iteration > 1000
                    x_min = iteration - 1000;
                    if get(chk_logX, 'Value') && x_min <= 0, x_min = 1; end
                    xlim(ax_time, [max(1, x_min), iteration]);
                end
                
                set(txt_status, 'String', sprintf('Active Particles: %d | Iteration: %d', length(p_x), iteration));
                drawnow; 
                
                % --- GIF FRAME CAPTURE ---
                if get(chk_recordGif, 'Value')
                    recorded_frames{end+1} = getframe(fig);
                    set(chk_recordGif, 'String', sprintf('Record GIF (Frames: %d)', length(recorded_frames)));
                end
            end
        else
            pause(0.1); 
        end
    end

    %% Callback Functions
    function toggleSim(~, ~)
        sim.isRunning = ~sim.isRunning;
        if sim.isRunning
            set(btn_toggle, 'String', 'PAUSE SIMULATION', 'BackgroundColor', [1 0.8 0.8]);
        else
            set(btn_toggle, 'String', 'START SIMULATION', 'BackgroundColor', [0.8 1 0.8]);
        end
    end

    function addDetector(~, ~)
        was_running = sim.isRunning; sim.isRunning = false;
        axes(ax_scatter); title(ax_scatter, 'CLICK ON PLOT TO PLACE PROBE...', 'Color', 'r');
        
        req_rad = str2double(get(edit_detRad, 'String'));
        if isnan(req_rad) || req_rad <= 0, req_rad = 0.05; end 
        
        req_ang_deg = str2double(get(edit_detAng, 'String'));
        if isnan(req_ang_deg), req_ang_deg = 0; end
        ang_rad = req_ang_deg * pi / 180;
        
        try
            [x, y] = ginput(1);
            if ~isempty(x) && ~isempty(y)
                c_idx = mod(length(detectors_Z), size(color_palette, 1)) + 1; det_color = color_palette(c_idx, :);
                
                px1 = x - req_rad * sin(ang_rad); py1 = y - req_rad * cos(ang_rad);
                px2 = x + req_rad * sin(ang_rad); py2 = y + req_rad * cos(ang_rad);
                m = plot(ax_scatter, [px1, px2], [py1, py2], '-', 'Color', det_color, 'LineWidth', 4);
                
                nx = x - 0.5 * req_rad * cos(ang_rad); ny = y + 0.5 * req_rad * sin(ang_rad);
                m_norm = plot(ax_scatter, [x, nx], [y, ny], '-', 'Color', det_color, 'LineWidth', 1.5);
                
                t = text(ax_scatter, x + req_rad + 0.02, y, '0', 'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold', 'BackgroundColor', det_color, 'Margin', 1);
                l = plot(ax_time, NaN, NaN, 'Color', det_color, 'LineWidth', 2);
                
                detectors_Z(end+1) = x; detectors_R(end+1) = y; detectors_rad(end+1) = req_rad; detectors_ang(end+1) = ang_rad; det_step_counts(end+1) = 0; 
                if isempty(time_history), det_counts_history{end+1} = []; else, det_counts_history{end+1} = NaN(1, length(time_history)); end
                
                h_det_markers(end+1) = m; h_det_normals(end+1) = m_norm; h_det_texts(end+1) = t; h_det_lines(end+1) = l;
            end
        catch
        end
        title(ax_scatter, 'Live Particle Flow (Blue: Primary, Red: CEX)', 'Color', 'k'); sim.isRunning = was_running;
    end

    function clearDetectors(~, ~)
        delete(h_det_markers); delete(h_det_normals); delete(h_det_texts); delete(h_det_lines);
        detectors_Z = []; detectors_R = []; detectors_rad = []; detectors_ang = []; det_step_counts = [];
        h_det_markers = gobjects(0); h_det_normals = gobjects(0); h_det_texts = gobjects(0); h_det_lines = gobjects(0); det_counts_history = {}; 
    end

    function updateScales(~, ~)
        if get(chk_logX, 'Value')
            set(ax_time, 'XScale', 'log'); xl = xlim(ax_time);
            if xl(1) <= 0, xlim(ax_time, [1, max(2, xl(2))]); end
        else
            set(ax_time, 'XScale', 'linear');
        end
        if get(chk_logY, 'Value')
            set(ax_time, 'YScale', 'log');
        else
            set(ax_time, 'YScale', 'linear');
        end
    end

    function exportData(~, ~)
        if isempty(time_history), msgbox('No data yet.', 'Export Failed', 'warn'); return; end
        was_running = sim.isRunning; if was_running, sim.isRunning = false; end
        
        [filename, pathname] = uiputfile('*.csv', 'Save Probe History');
        if ~isequal(filename, 0) && ~isequal(pathname, 0)
            full_path = fullfile(pathname, filename);
            try
                export_matrix = time_history'; var_names = {'Iteration'};
                for i = 1:length(det_counts_history)
                    export_matrix = [export_matrix, det_counts_history{i}']; var_names{end+1} = sprintf('Probe_%d', i);
                end
                T = array2table(export_matrix, 'VariableNames', var_names);
                writetable(T, full_path); msgbox('Export Saved!', 'Success', 'help');
            catch ME
                errordlg(ME.message, 'Export Error');
            end
        end
        if was_running, sim.isRunning = true; end
    end

    function saveGif(~, ~)
        if isempty(recorded_frames)
            msgbox('No frames recorded! Check "Record GIF" and run the simulation first.', 'Error', 'error'); return;
        end
        
        was_running = sim.isRunning;
        if was_running
            sim.isRunning = false; set(btn_toggle, 'String', 'START SIMULATION', 'BackgroundColor', [0.8 1 0.8]);
        end
        
        [filename, pathname] = uiputfile('*.gif', 'Save Animation');
        if ~isequal(filename, 0) && ~isequal(pathname, 0)
            full_path = fullfile(pathname, filename);
            h_wait = waitbar(0, 'Generating GIF... Please wait.');
            
            try
                for idx = 1:length(recorded_frames)
                    im = frame2im(recorded_frames{idx});
                    [imind, cm] = rgb2ind(im, 256);
                    if idx == 1
                        imwrite(imind, cm, full_path, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
                    else
                        imwrite(imind, cm, full_path, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
                    end
                    waitbar(idx/length(recorded_frames), h_wait);
                end
                close(h_wait);
                msgbox('GIF Saved Successfully!', 'Success', 'help');
                
                % Clear memory after saving
                recorded_frames = {};
                set(chk_recordGif, 'Value', 0);
                set(chk_recordGif, 'String', 'Record GIF (Frames: 0)');
            catch ME
                close(h_wait); errordlg(['Failed to save GIF: ', ME.message], 'Error');
            end
        end
        if was_running
            sim.isRunning = true; set(btn_toggle, 'String', 'PAUSE SIMULATION', 'BackgroundColor', [1 0.8 0.8]);
        end
    end

    function onClose(~, ~)
        sim.isRunning = false; delete(gcf);
    end
end