function DigitalTwin_Optimized()
    % --- 1. GLOBAL STATE & SIMULATION VARIABLES ---
    sim.isRunning = false;
    sim.dt = 2e-9; 
    sim.q = 1.602e-19;
    sim.m_XE = 131.293 * 1.6605e-27;
    sim.kB = 1.380649e-23; 
    
    % Vectorized Particle Arrays
    p_x = []; p_y = []; p_vx = []; p_vy = []; p_isCEX = false(0);
    
    % Mesh, Field, and Damage variables
    X = []; Y = []; V = []; Ex = []; Ey = []; 
    isBound = []; V_fixed = []; damage_map = [];
    Lx = 8; Ly = 4; dx = 0.05; dy = 0.05; 
    
    % Telemetry & 3D Objects
    iter_history = []; ebs_history = []; div_history = [];
    recorded_frames = {};
    f3d = []; ax3d = []; h_surf3d = []; h_prim3d = []; h_cex3d = [];
    
    % --- 2. SETUP THE GUI (Ultra-Scaled for 125/150% Displays) ---
    % Width set to 1280 to prevent right-side clipping
    fig = uifigure('Name', '2-Grid Digital Twin: Morphing, Telemetry & LIVE 3D', 'Position', [20, 30, 1280, 780], 'Color', '#F4F6F9', 'CloseRequestFcn', @onClose);
    
    % Control Panel Container
    pnl = uipanel(fig, 'Position', [10, 10, 300, 760], 'BackgroundColor', '#FFFFFF');
    
    % --- 2A. ELECTRICAL & GEOMETRY ---
    ypos = 730; step = 26;
    uilabel(pnl, 'Text', '1. GRID DESIGN', 'Position', [10, ypos, 280, 22], 'FontWeight', 'bold', 'FontColor', [0.1 0.3 0.6]); ypos = ypos - step;
    uilabel(pnl, 'Text', 'Screen Volts (V):', 'Position', [10, ypos, 130, 22]); ui_Vscreen = uieditfield(pnl, 'numeric', 'Position', [150, ypos, 130, 22], 'Value', 1000); ypos = ypos - step;
    uilabel(pnl, 'Text', 'Accel Volts (V):', 'Position', [10, ypos, 130, 22]); ui_Vaccel = uieditfield(pnl, 'numeric', 'Position', [150, ypos, 130, 22], 'Value', -200); ypos = ypos - step;
    uilabel(pnl, 'Text', 'Grid Gap (mm):', 'Position', [10, ypos, 130, 22], 'FontWeight', 'bold'); ui_Gap = uieditfield(pnl, 'numeric', 'Position', [150, ypos, 130, 22], 'Value', 1.0); ypos = ypos - step;
    uilabel(pnl, 'Text', 'Screen Thick (mm):', 'Position', [10, ypos, 130, 22]); ui_Tscreen = uieditfield(pnl, 'numeric', 'Position', [150, ypos, 130, 22], 'Value', 0.6); ypos = ypos - step;
    uilabel(pnl, 'Text', 'Accel Thick (mm):', 'Position', [10, ypos, 130, 22]); ui_Taccel = uieditfield(pnl, 'numeric', 'Position', [150, ypos, 130, 22], 'Value', 1.2); ypos = ypos - step;
    uilabel(pnl, 'Text', 'Screen Hole R (mm):', 'Position', [10, ypos, 130, 22]); ui_RadS = uieditfield(pnl, 'numeric', 'Position', [150, ypos, 130, 22], 'Value', 1.0); ypos = ypos - step;
    uilabel(pnl, 'Text', 'Accel Hole R (mm):', 'Position', [10, ypos, 130, 22]); ui_RadA = uieditfield(pnl, 'numeric', 'Position', [150, ypos, 130, 22], 'Value', 0.8); ypos = ypos - step;
    uilabel(pnl, 'Text', 'Screen Chamfer (°):', 'Position', [10, ypos, 130, 22]); ui_ChamferS = uieditfield(pnl, 'numeric', 'Position', [150, ypos, 130, 22], 'Value', 0); ypos = ypos - step;
    uilabel(pnl, 'Text', 'Accel Chamfer (°):', 'Position', [10, ypos, 130, 22]); ui_ChamferA = uieditfield(pnl, 'numeric', 'Position', [150, ypos, 130, 22], 'Value', 15); ypos = ypos - step - 5;

    % --- 2B. PLASMA & MORPHING ---
    uilabel(pnl, 'Text', '2. PLASMA & MORPHING', 'Position', [10, ypos, 280, 22], 'FontWeight', 'bold', 'FontColor', [0.1 0.3 0.6]); ypos = ypos - step;
    uilabel(pnl, 'Text', 'Ion Temp (eV):', 'Position', [10, ypos, 130, 22], 'FontColor', 'b'); ui_Ti = uieditfield(pnl, 'numeric', 'Position', [150, ypos, 130, 22], 'Value', 2.0); ypos = ypos - step;
    uilabel(pnl, 'Text', 'Neutral Temp (K):', 'Position', [10, ypos, 130, 22], 'FontColor', '#D95319'); ui_Tn = uieditfield(pnl, 'numeric', 'Position', [150, ypos, 130, 22], 'Value', 300); ypos = ypos - step;
    uilabel(pnl, 'Text', 'Neutral Dens n0:', 'Position', [10, ypos, 130, 22]); ui_n0 = uieditfield(pnl, 'numeric', 'Position', [150, ypos, 130, 22], 'Value', 1e20); ypos = ypos - step;
    uilabel(pnl, 'Text', 'Accel. Factor (X):', 'Position', [10, ypos, 130, 22], 'FontColor', 'r'); ui_Accel = uieditfield(pnl, 'numeric', 'Position', [150, ypos, 130, 22], 'Value', 5e13); ypos = ypos - step;
    uilabel(pnl, 'Text', 'Cell Fail Thresh:', 'Position', [10, ypos, 130, 22], 'FontColor', 'r'); ui_Thresh = uieditfield(pnl, 'numeric', 'Position', [150, ypos, 130, 22], 'Value', 1.0); ypos = ypos - 15;

    % --- 2C. EXECUTION BUTTONS ---
    btn_recalc = uibutton(pnl, 'Text', '1. BUILD DOMAIN', 'Position', [10, ypos-25, 280, 35], 'BackgroundColor', [1 0.9 0.6], 'FontWeight', 'bold', 'ButtonPushedFcn', @buildDomain);
    btn_toggle = uibutton(pnl, 'Text', '2. START BEAM EXTRACTION', 'Position', [10, ypos-65, 280, 35], 'BackgroundColor', [0.8 1 0.8], 'FontWeight', 'bold', 'ButtonPushedFcn', @toggleSim);
    
    btn_3d = uibutton(pnl, 'Text', 'OPEN LIVE 3D CAD VIEW', 'Position', [10, ypos-105, 280, 35], 'BackgroundColor', '#E0F7FA', 'FontWeight', 'bold', 'ButtonPushedFcn', @init3D);
    
    chk_recordGif = uicheckbox(pnl, 'Text', 'Record Frames (0)', 'Position', [15, ypos-135, 200, 22], 'FontWeight', 'bold', 'FontColor', 'r');
    btn_saveGif = uibutton(pnl, 'Text', 'Save GIF Animation', 'Position', [10, ypos-165, 280, 25], 'BackgroundColor', [0.9 0.8 1], 'ButtonPushedFcn', @saveGif);
    
    txt_status = uilabel(pnl, 'Text', 'Status: Ready.', 'Position', [10, 5, 280, 30], 'FontSize', 10, 'FontWeight', 'bold', 'FontColor', [0 0.5 0]);

    % --- RIGHT PANEL: AXES LAYOUT ---
    % Top: Live Morphing Extraction (Starts at X=320, width 940)
    ax_live = uiaxes(fig, 'Position', [320, 380, 940, 390]);
    title(ax_live, 'Live Axisymmetric Plasma Extraction & E-Field');
    xlabel(ax_live, 'Axial Position X (mm)'); ylabel(ax_live, 'Radial Position Y (mm)');
    hold(ax_live, 'on'); box(ax_live, 'on'); axis(ax_live, 'equal'); xlim(ax_live, [0, Lx]); ylim(ax_live, [0, Ly]);
    
    h_contour = []; h_bound = [];
    h_beam = scatter(ax_live, [], [], 2, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
    h_cex  = scatter(ax_live, [], [], 5, 'r', 'filled');

    % Bottom Row: 3 Plots perfectly distributed across 940 pixels
    % Widths = 300, Gaps = 20
    
    % Bottom Left: Sputter Damage Map
    ax_damage = uiaxes(fig, 'Position', [320, 10, 300, 320]);
    title(ax_damage, 'Live Sputter Damage Map');
    xlabel(ax_damage, 'Axial X (mm)'); ylabel(ax_damage, 'Radial Y (mm)');
    hold(ax_damage, 'on'); box(ax_damage, 'on');
    colormap(ax_damage, flipud(hot(256))); 
    
    % Bottom Mid: EBS Telemetry
    ax_ebs = uiaxes(fig, 'Position', [640, 10, 300, 320]);
    title(ax_ebs, 'Electron Backstreaming (EBS)');
    xlabel(ax_ebs, 'Iteration'); ylabel(ax_ebs, 'Min Centerline Volts (V)');
    hold(ax_ebs, 'on'); grid(ax_ebs, 'on');
    h_ebs = plot(ax_ebs, NaN, NaN, 'm-', 'LineWidth', 2);
    yline(ax_ebs, -20, 'r--', 'EBS Failure Limit (-20V)', 'LabelHorizontalAlignment', 'left');

    % Bottom Right: Divergence Telemetry (Right edge = 960+300 = 1260 < 1280)
    ax_div = uiaxes(fig, 'Position', [960, 10, 300, 320]);
    title(ax_div, 'Primary Beam 95% Divergence');
    xlabel(ax_div, 'Iteration'); ylabel(ax_div, 'Half-Angle (Deg)');
    hold(ax_div, 'on'); grid(ax_div, 'on');
    h_div = plot(ax_div, NaN, NaN, 'b-', 'LineWidth', 2);

    % --- 3. MAIN SIMULATION LOOP ---
    iteration = 0;
    while ishandle(fig)
        if sim.isRunning && ~isempty(Ex)
            iteration = iteration + 1;
            
            % 1. Inject Primaries
            num_inject = 30; 
            new_y = linspace(0.05, ui_RadS.Value - 0.1, num_inject)' + (rand(num_inject,1)-0.5)*0.05;
            new_x = repmat(0.1, num_inject, 1);
            
            v_bohm = sqrt(2 * sim.q * 50 / sim.m_XE); 
            v_spread = sqrt(sim.q * ui_Ti.Value / sim.m_XE); 
            
            p_x = [p_x; new_x]; p_y = [p_y; new_y];
            p_vx = [p_vx; repmat(v_bohm, num_inject, 1) + randn(num_inject,1)*v_spread]; 
            p_vy = [p_vy; randn(num_inject,1)*v_spread];
            p_isCEX = [p_isCEX; false(num_inject, 1)];
            
            % 2. Push Particles
            Ex_p = interp2(X, Y, Ex, p_x, p_y, 'linear', 0);
            Ey_p = interp2(X, Y, Ey, p_x, p_y, 'linear', 0);
            
            p_vx = p_vx + (sim.q / sim.m_XE) .* Ex_p .* sim.dt;
            p_vy = p_vy + (sim.q / sim.m_XE) .* Ey_p .* sim.dt;
            p_x = p_x + p_vx .* sim.dt .* 1000;
            p_y = p_y + p_vy .* sim.dt .* 1000;
            
            % 3. Extract Telemetry
            max_grid_x = 1.0 + ui_Tscreen.Value + ui_Gap.Value + ui_Taccel.Value;
            post_grid_mask = (~p_isCEX) & (p_x > max_grid_x);
            if sum(post_grid_mask) > 5
                angles = abs(atan(p_vy(post_grid_mask) ./ p_vx(post_grid_mask))) * 180/pi;
                current_div = prctile(angles, 95);
            else
                current_div = NaN; 
            end
            
            min_pot = min(V(1, :));
            
            % 4. Hit Detection
            hit_grid = interp2(X, Y, double(isBound), p_x, p_y, 'nearest', 1) == 1;
            out_of_bounds = p_x < 0 | p_x > Lx | p_y < 0 | p_y > Ly | isnan(p_x);
            
            % 5. ENERGY-DEPENDENT SPUTTERING & MORPHING
            is_erosion_hit = hit_grid & p_isCEX;
            if any(is_erosion_hit)
                hx = p_x(is_erosion_hit); hy = p_y(is_erosion_hit);
                hvx = p_vx(is_erosion_hit); hvy = p_vy(is_erosion_hit);
                
                ix = min(size(X,2), max(1, round(hx / dx) + 1));
                iy = min(size(Y,1), max(1, round(hy / dy) + 1));
                
                v_mag_sq = hvx.^2 + hvy.^2;
                E_eV = (0.5 * sim.m_XE * v_mag_sq) / sim.q;
                
                Y_yield = zeros(size(E_eV));
                valid_E = E_eV > 30;
                Y_yield(valid_E) = 1.05e-4 .* (E_eV(valid_E) - 30).^1.5; 
                
                damage_added = accumarray([iy, ix], Y_yield .* ui_Accel.Value, size(damage_map));
                damage_map = damage_map + damage_added;
                
                % Structural failure check
                broken_cells = (damage_map > ui_Thresh.Value) & isBound;
                if any(broken_cells, 'all')
                    isBound(broken_cells) = false; 
                    damage_map(broken_cells) = 0; 
                    
                    txt_status.Text = 'Cell Failed! Remeshing Laplace...'; drawnow;
                    recalcLaplace();
                    update3DSurface(); 
                    txt_status.Text = sprintf('Active Particles: %d | Iteration: %d', length(p_x), iteration);
                end
            end
            
            % Purge dead
            dead_mask = hit_grid | out_of_bounds;
            p_x(dead_mask) = []; p_y(dead_mask) = [];
            p_vx(dead_mask) = []; p_vy(dead_mask) = []; p_isCEX(dead_mask) = [];
            
            % 6. Charge Exchange (MCC)
            primary_mask = ~p_isCEX & (p_x > 1.0); %%This defines the axial distance after which charge exhange occurs
            if any(primary_mask)
                v_mag = sqrt(p_vx(primary_mask).^2 + p_vy(primary_mask).^2);
                g = max(v_mag, 1);
                sigma = ((-0.8821 .* log(g) + 15.1262).^2) .* 1e-20;
                prob = 1 - exp(-ui_n0.Value .* sigma .* g .* sim.dt);
                
                collided = rand(sum(primary_mask), 1) < (prob .* 1); %%Multipy probability by 100 to speed up visualizaiton
                if any(collided)
                    idx = find(primary_mask); c_idx = idx(collided);
                    num_cex = length(c_idx);
                    neut_vth = sqrt(2 * sim.kB * ui_Tn.Value / sim.m_XE); 
                    p_vx(c_idx) = randn(num_cex,1) .* neut_vth; 
                    p_vy(c_idx) = randn(num_cex,1) .* neut_vth;
                    p_isCEX(c_idx) = true; 
                end
            end
            
            % 7. GUI, Telemetry & LIVE 3D Updates
            if mod(iteration, 5) == 0
                set(h_beam, 'XData', p_x(~p_isCEX), 'YData', p_y(~p_isCEX));
                set(h_cex, 'XData', p_x(p_isCEX), 'YData', p_y(p_isCEX));
                
                % Telemetry
                iter_history = [iter_history, iteration];
                ebs_history = [ebs_history, min_pot];
                div_history = [div_history, current_div];
                
                set(h_ebs, 'XData', iter_history, 'YData', ebs_history);
                set(h_div, 'XData', iter_history, 'YData', div_history);
                xlim(ax_ebs, [max(0, iteration-400), max(100, iteration)]);
                xlim(ax_div, [max(0, iteration-400), max(100, iteration)]);
                
                % Damage Map
                cla(ax_damage);
                contourf(ax_damage, X, Y, damage_map, 15, 'LineStyle', 'none');
                [gy, gx] = find(isBound);
                scatter(ax_damage, (gx-1)*dx, (gy-1)*dy, 2, [0.3 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.5);
                xlim(ax_damage, [0.5, Lx]); ylim(ax_damage, [0, Ly]);
                
                % --- UPDATE LIVE 3D VIEW ---
                if ~isempty(f3d) && isvalid(f3d)
                    prim_mask = ~p_isCEX;
                    if any(prim_mask)
                        th_p = rand(sum(prim_mask), 1) * 2 * pi;
                        set(h_prim3d, 'XData', p_x(prim_mask), 'YData', p_y(prim_mask).*cos(th_p), 'ZData', p_y(prim_mask).*sin(th_p));
                    end
                    cex_mask = p_isCEX;
                    if any(cex_mask)
                        th_c = rand(sum(cex_mask), 1) * 2 * pi;
                        set(h_cex3d, 'XData', p_x(cex_mask), 'YData', p_y(cex_mask).*cos(th_c), 'ZData', p_y(cex_mask).*sin(th_c));
                    end
                    if mod(iteration, 20) == 0
                        update3DSurfaceColor();
                    end
                end
                
                if chk_recordGif.Value
                    recorded_frames{end+1} = getframe(fig);
                    chk_recordGif.Text = sprintf('Record Frames (%d)', length(recorded_frames));
                end
                drawnow;
            end
        else
            pause(0.1);
        end
    end

    % --- CALLBACK FUNCTIONS ---
    function toggleSim(~, ~)
        if isempty(Ex), uialert(fig, 'Build Domain first!', 'Warning'); return; end
        sim.isRunning = ~sim.isRunning;
        if sim.isRunning
            btn_toggle.Text = '2. PAUSE BEAM EXTRACTION'; btn_toggle.BackgroundColor = [1 0.8 0.8];
        else
            btn_toggle.Text = '2. RESUME BEAM EXTRACTION'; btn_toggle.BackgroundColor = [0.8 1 0.8];
        end
    end

    function buildDomain(~, ~)
        sim.isRunning = false;
        btn_toggle.Text = '2. START BEAM EXTRACTION'; btn_toggle.BackgroundColor = [0.8 1 0.8];
        p_x = []; p_y = []; p_vx = []; p_vy = []; p_isCEX = false(0);
        iter_history = []; ebs_history = []; div_history = [];
        
        txt_status.Text = 'Building 2-Grid Domain...'; drawnow;
        
        nx = round(Lx/dx) + 1; ny = round(Ly/dy) + 1;
        [X, Y] = meshgrid(linspace(0, Lx, nx), linspace(0, Ly, ny));
        V = zeros(ny, nx); isBound = false(ny, nx); V_fixed = zeros(ny, nx);
        damage_map = zeros(ny, nx);
        
        Vs = ui_Vscreen.Value; Va = ui_Vaccel.Value;
        ts = ui_Tscreen.Value; ta = ui_Taccel.Value; gap = ui_Gap.Value;
        rs = ui_RadS.Value; ra = ui_RadA.Value;
        cham_s = ui_ChamferS.Value; cham_a = ui_ChamferA.Value;
        
        screen_start = 1.0; screen_end = screen_start + ts;
        accel_start = screen_end + gap; accel_end = accel_start + ta;
        
        in_screen = X >= screen_start & X <= screen_end;
        in_accel  = X >= accel_start & X <= accel_end;
        
        R_screen = rs + max(0, (X - screen_start)) * tand(cham_s);
        R_accel  = ra + max(0, (X - accel_start)) * tand(cham_a);
        
        mask_s = in_screen & Y >= R_screen;
        isBound(mask_s) = true; V_fixed(mask_s) = Vs;
        
        mask_a = in_accel & Y >= R_accel;
        isBound(mask_a) = true; V_fixed(mask_a) = Va;
        
        isBound(:, 1) = true; V_fixed(:, 1) = Vs + 50; 
        
        recalcLaplace();
        update3DSurface(); 
    end

    function recalcLaplace()
        V(isBound) = V_fixed(isBound);
        for i = 1:500 
            V(2:end-1, 2:end-1) = 0.25*(V(3:end, 2:end-1) + V(1:end-2, 2:end-1) + V(2:end-1, 3:end) + V(2:end-1, 1:end-2));
            V(isBound) = V_fixed(isBound); V(1, :) = V(2, :); V(end, :) = V(end-1, :); V(:, end) = V(:, end-1);
        end
        [Ex, Ey] = gradient(-V, dx*1e-3, dy*1e-3);
        
        cla(ax_live);
        [~, h_contour] = contourf(ax_live, X, Y, V, 20, 'LineStyle', 'none'); colormap(ax_live, turbo);
        [gy, gx] = find(isBound);
        h_bound = scatter(ax_live, (gx-1)*dx, (gy-1)*dy, 12, 'k', 'filled', 'MarkerFaceAlpha', 0.8);
        h_beam = scatter(ax_live, [], [], 2, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
        h_cex  = scatter(ax_live, [], [], 5, 'r', 'filled');
        
        txt_status.Text = 'Domain Ready.';
    end

    % --- THE LIVE 3D RENDERING ENGINE ---
    function init3D(~, ~)
        if isempty(isBound)
            uialert(fig, 'Build domain first to generate 3D model.', 'Warning'); return;
        end
        
        if isempty(f3d) || ~isvalid(f3d)
            f3d = figure('Name', 'Live 3D Particle & Morphing Viewer', 'Color', 'k', 'Position', [200, 200, 900, 700]);
            ax3d = axes(f3d);
            hold(ax3d, 'on');
            
            % Init empty particles
            h_prim3d = scatter3(ax3d, NaN, NaN, NaN, 4, [0.3 0.6 1], 'filled', 'MarkerFaceAlpha', 0.6);
            h_cex3d  = scatter3(ax3d, NaN, NaN, NaN, 8, [1 0.2 0.2], 'filled', 'MarkerFaceAlpha', 0.8);
            
            % Fix axis limits so the camera doesn't jump
            xlim(ax3d, [0, Lx]);
            ylim(ax3d, [-Ly, Ly]);
            zlim(ax3d, [-Ly, Ly]);
            axis(ax3d, 'manual'); 
            
            camlight(ax3d, 'headlight');
            lighting(ax3d, 'gouraud');
            material(ax3d, 'dull'); 
            
            set(ax3d, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
            title(ax3d, 'Live 3D Extraction & Surface Sputtering', 'Color', 'w');
            xlabel(ax3d, 'Axial Axis X (mm)'); ylabel(ax3d, 'Radial Y (mm)'); zlabel(ax3d, 'Radial Z (mm)');
            
            % View set to R-Z plane (Looking down the Y axis to see X and Z)
            view(ax3d, [0, 0]); 
            grid(ax3d, 'on');
            
            update3DSurface();
        else
            figure(f3d); 
        end
    end

    function update3DSurface()
        if isempty(f3d) || ~isvalid(f3d), return; end
        
        if ~isempty(h_surf3d) && isvalid(h_surf3d)
            delete(h_surf3d); 
        end
        
        x_range = X(1, :);
        R_profile = zeros(size(x_range));
        D_profile = zeros(size(x_range));
        
        for i = 1:length(x_range)
            solid_y = find(isBound(:, i));
            if ~isempty(solid_y)
                iy = solid_y(1);
                R_profile(i) = Y(iy, 1);
                D_profile(i) = damage_map(iy, i);
            else
                R_profile(i) = NaN;
                D_profile(i) = NaN;
            end
        end
        
        theta_surf = linspace(0, 2*pi, 60);
        [Theta_mat, X_mat] = meshgrid(theta_surf, x_range);
        R_mat = repmat(R_profile', 1, length(theta_surf));
        C_mat = repmat(D_profile', 1, length(theta_surf));
        
        Y_mat = R_mat .* cos(Theta_mat);
        Z_mat = R_mat .* sin(Theta_mat);
        
        h_surf3d = surf(ax3d, X_mat, Y_mat, Z_mat, C_mat, 'EdgeColor', 'none');
        colormap(ax3d, flipud(hot(256))); 
        uistack(h_prim3d, 'top'); uistack(h_cex3d, 'top');
    end

    function update3DSurfaceColor()
        if isempty(f3d) || ~isvalid(f3d) || isempty(h_surf3d), return; end
        x_range = X(1, :);
        D_profile = zeros(size(x_range));
        for i = 1:length(x_range)
            solid_y = find(isBound(:, i));
            if ~isempty(solid_y)
                D_profile(i) = damage_map(solid_y(1), i);
            else
                D_profile(i) = NaN;
            end
        end
        C_mat = repmat(D_profile', 1, 60);
        set(h_surf3d, 'CData', C_mat);
    end

    function saveGif(~, ~)
        if isempty(recorded_frames), uialert(fig, 'No frames recorded!', 'Error'); return; end
        was_running = sim.isRunning; sim.isRunning = false;
        [filename, pathname] = uiputfile('*.gif', 'Save Animation');
        if ~isequal(filename, 0)
            full_path = fullfile(pathname, filename);
            for idx = 1:length(recorded_frames)
                im = frame2im(recorded_frames{idx}); [imind, cm] = rgb2ind(im, 256);
                if idx == 1, imwrite(imind, cm, full_path, 'gif', 'Loopcount', inf, 'DelayTime', 0.05);
                else, imwrite(imind, cm, full_path, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05); end
            end
            recorded_frames = {}; chk_recordGif.Value = 0; chk_recordGif.Text = 'Record Frames (0)';
            uialert(fig, 'GIF Saved Successfully!', 'Success');
        end
        sim.isRunning = was_running;
    end

    function onClose(~, ~)
        sim.isRunning = false; 
        if ~isempty(f3d) && isvalid(f3d), delete(f3d); end
        delete(fig); 
    end
end
