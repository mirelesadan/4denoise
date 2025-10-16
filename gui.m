%%%%%%%%%%%%%%%%%
% Load Datasets %
%%%%%%%%%%%%%%%%%

% Height Map
load('height_map.mat');

% Uncorrected Strain Maps
load('exx_rippleData.mat');
load('eyy_rippleData.mat');
load('exy_rippleData.mat');
load('erot_rippleData.mat');

% Corrected Strain Maps (erot does not change)
load('exx_rippleData_corrected.mat');
load('eyy_rippleData_corrected.mat');
load('exy_rippleData_corrected.mat');

% Additional Datasets
load('haadf.mat');   % 2D HAADF image
load('tilts.mat');   % 256×256×2: tilts(:,:,1)=phase ∈ [0,2π], tilts(:,:,2)=amplitude ∈ [0,1]



%%%%
% 1) Compute gradients of the height map
[dh_dy, dh_dx] = gradient(height_map);

% 2) Compute phase as angle of the gradient, mapped into [0,2π)
phase_new = atan2(dh_dy, dh_dx);      % returns values in (–π, π]
phase_new = mod(phase_new, 2*pi);     % now in [0, 2π)

% 3) Replace the phase slice in `tilts`
tilts(:,:,1) = phase_new;
%%%%


% Define cropping fractions
y_min_frac = 0.1;
y_max_frac = 0.8;
x_min_frac = 0.1;
x_max_frac = 0.8;

% List of variables to crop in base workspace
variables_to_crop = { ...
    'height_map', ...
    'exx_rippleData', 'eyy_rippleData', ...
    'exy_rippleData', 'erot_rippleData', ...
    'exx_rippleData_corrected', 'eyy_rippleData_corrected', ...
    'exy_rippleData_corrected' ...
};

% Crop each named 2D variable to its central 80%
for i = 1:numel(variables_to_crop)
    var_name = variables_to_crop{i};
    data = evalin('base', var_name);
    [ny, nx] = size(data);
    y_start = round(y_min_frac * ny);
    y_end   = round(y_max_frac * ny);
    x_start = round(x_min_frac * nx);
    x_end   = round(x_max_frac * nx);
    cropped_data = data(y_start:y_end, x_start:x_end);
    assignin('base', var_name, cropped_data);
end

% Crop HAADF
haadf = evalin('base', 'haadf');
[ny_h, nx_h] = size(haadf);
hs = round(y_min_frac * ny_h);
he = round(y_max_frac * ny_h);
ws = round(x_min_frac * nx_h);
we = round(x_max_frac * nx_h);
haadf_cropped = haadf(hs:he, ws:we);
assignin('base', 'haadf', haadf_cropped);

% Crop tilts (3D array: ny_t × nx_t × ntilt)
tilts = evalin('base', 'tilts');
[ny_t, nx_t, ~] = size(tilts);
ts = round(y_min_frac * ny_t);
te = round(y_max_frac * ny_t);
us = round(x_min_frac * nx_t);
ve = round(x_max_frac * nx_t);
tilts_cropped = tilts(ts:te, us:ve, :);
assignin('base', 'tilts', tilts_cropped);


%%%%%%%%%%%%%%
% Initial GUI %
%%%%%%%%%%%%%%

initial_stretch_factor = 1;
cubeSize = 10;
cubeColor = [0.5, 0.5, 0.5];

% Flipped topology for Z data
topology = height_map * initial_stretch_factor;
topology_flipped = flipud(topology);

% Default surface_color: ε_rot
surface_color = erot_rippleData;
surface_color_flipped = flipud(surface_color);
color_limits = [-abs(max(surface_color_flipped(:))), abs(max(surface_color_flipped(:)))];

fig = figure('Name', 'Three-Dimensional Strain & Overlay Profiling', ...
             'Units', 'pixels', 'Position', [0, 0, 1000, 660], 'Color', 'w');

% Store data in appdata for callbacks
setappdata(fig, 'h_surf', []);
setappdata(fig, 'erot_rippleData', erot_rippleData);
setappdata(fig, 'exx_rippleData', exx_rippleData);
setappdata(fig, 'eyy_rippleData', eyy_rippleData);
setappdata(fig, 'exy_rippleData', exy_rippleData);
setappdata(fig, 'exx_rippleData_corrected', exx_rippleData_corrected);
setappdata(fig, 'eyy_rippleData_corrected', eyy_rippleData_corrected);
setappdata(fig, 'exy_rippleData_corrected', exy_rippleData_corrected);
setappdata(fig, 'height_map', height_map);
setappdata(fig, 'haadf', haadf);
setappdata(fig, 'tilts', tilts);
setappdata(fig, 'lastChoice', 'ε_rot');  % Track last selected display

% Main surface
h_surf = surf(topology_flipped, surface_color_flipped);
shading interp;
setappdata(fig, 'h_surf', h_surf);

% compute tilt angles (degrees) from the (cropped) height_map ---
[dh_dy2, dh_dx2] = gradient(height_map);
tilt_angles      = atand( sqrt(dh_dy2.^2 + dh_dx2.^2) );
maxTilt          = max(tilt_angles(:));

% put it into the GUI’s appdata
setappdata(0, 'maxTilt', maxTilt);

% Default colormap: RdBu for strain/rotation
colormap(slanCM('RdBu', 256));
caxis(color_limits);
hold on;

% Coordinate axes
axis_length = 25;
origin = [-axis_length, -axis_length, 0];
h_quiver_x = quiver3(origin(1), origin(2), origin(3), axis_length, 0, 0, 'r', 'LineWidth', 2);
h_quiver_y = quiver3(origin(1), origin(2), origin(3), 0, axis_length, 0, 'g', 'LineWidth', 2);
h_quiver_z = quiver3(origin(1), origin(2), origin(3), 0, 0, axis_length, 'b', 'LineWidth', 2);
h_text_x = text(origin(1)+axis_length, origin(2), 0,            'X', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'r');
h_text_y = text(origin(1), origin(2)+axis_length, 0,            'Y', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'g');
h_text_z = text(origin(1), origin(2), origin(3)+axis_length, 'Z', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'b');

% Initial cube
createCube(cubeSize, initial_stretch_factor, cubeColor, height_map);

axis off; axis vis3d; axis equal;
alpha(0.95);

%%%%%%%%%%%%
% Colorbar %
%%%%%%%%%%%%

cb = colorbar('Tag', 'MainColorbar');
title(cb, 'Rotation (°)');
cb.FontSize = 14;
cb.Title.FontSize = 18;
cb.Position = [0.9, 0.3, 0.02, 0.4];

% ±Max controls (visible for strain)
text_pm = uicontrol('Style', 'text', 'String', '± Max', 'Units', 'normalized', ...
                    'Position', [0.88, 0.25, 0.08, 0.03], 'FontSize', 10);
edit_pm = uicontrol('Style', 'edit', 'String', num2str(color_limits(2)), 'Units', 'normalized', ...
                    'Position', [0.88, 0.22, 0.08, 0.03], 'FontSize', 10, ...
                    'Callback', @(src, event) updateColorLimitsSymmetric(src));
                
% Min/Max controls (visible for HAADF/Height)
textMin = uicontrol('Style', 'text', 'String', 'Min', 'Units', 'normalized', ...
                    'Position', [0.88, 0.25, 0.08, 0.03], 'FontSize', 10, 'Visible', 'off');
editMin = uicontrol('Style', 'edit', 'String', num2str(min(surface_color_flipped(:))), 'Units', 'normalized', ...
                    'Position', [0.88, 0.22, 0.08, 0.03], 'FontSize', 10, 'Visible', 'off', ...
                    'Callback', @(src, event) updateColorLimitsSeparate());
                
textMax = uicontrol('Style', 'text', 'String', 'Max', 'Units', 'normalized', ...
                    'Position', [0.88, 0.18, 0.08, 0.03], 'FontSize', 10, 'Visible', 'off');
editMax = uicontrol('Style', 'edit', 'String', num2str(max(surface_color_flipped(:))), 'Units', 'normalized', ...
                    'Position', [0.88, 0.15, 0.08, 0.03], 'FontSize', 10, 'Visible', 'off', ...
                    'Callback', @(src, event) updateColorLimitsSeparate());

setappdata(fig, 'text_pm', text_pm);
setappdata(fig, 'edit_pm', edit_pm);
setappdata(fig, 'textMin', textMin);
setappdata(fig, 'editMin', editMin);
setappdata(fig, 'textMax', textMax);
setappdata(fig, 'editMax', editMax);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Non-Strain Display Radio Group %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

groupNonStrain = uibuttongroup('Position', [0.07, 0.46, 0.15, 0.14], 'Units', 'normalized', ...
    'Title', 'Overlay', 'FontSize', 10, ...
    'SelectionChangedFcn', @(src, event) nonStrainChanged(event));

uicontrol(groupNonStrain, 'Style', 'radiobutton', 'String', 'HAADF',     'FontSize', 10, 'Position', [10, 50,  70, 20]);
uicontrol(groupNonStrain, 'Style', 'radiobutton', 'String', 'Height',    'FontSize', 10, 'Position', [90, 50,  70, 20]);
uicontrol(groupNonStrain, 'Style', 'radiobutton', 'String', 'Phase Map', 'FontSize', 10, 'Position', [10, 20,  85, 20]);

setappdata(fig, 'groupNonStrain', groupNonStrain);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Interpolated Shading Toggle
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
interpToggle = uicontrol('Style','togglebutton', ...
    'String','Interp On', ...
    'Units','normalized', ...
    'Position',[0.07, 0.61, 0.15, 0.05], ...
    'Value',1, ...             % default = on
    'FontSize',10, ...
    'Callback',@(src,~) updateShading(src));
setappdata(fig,'interpToggle',interpToggle);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Strain Correction Toggle %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

strainCorrectionToggle = uicontrol('Style', 'togglebutton', 'String', 'Toggle Strain Correction', ...
                                   'FontSize', 10, 'Units', 'normalized', ...
                                   'Position', [0.07, 0.4, 0.15, 0.05], 'Value', 0, ...
                                   'Callback', @(src, event) updateDisplay());

setappdata(fig, 'strainCorrectionToggle', strainCorrectionToggle);

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Strain Display Group   %
%%%%%%%%%%%%%%%%%%%%%%%%%%

groupStrain = uibuttongroup('Position', [0.07, 0.26, 0.15, 0.12], 'Units', 'normalized', ...
    'Title', 'Strain', 'FontSize', 10, ...
    'SelectionChangedFcn', @(src, event) strainChanged(event));

uicontrol(groupStrain, 'Style', 'radiobutton', 'String', 'ε_rot', 'FontSize', 10, 'Position', [10, 47.2,  70, 20], 'Value', 1);
uicontrol(groupStrain, 'Style', 'radiobutton', 'String', 'ε_xx',  'FontSize', 10, 'Position', [10, 13.6,  70, 20]);
uicontrol(groupStrain, 'Style', 'radiobutton', 'String', 'ε_yy',  'FontSize', 10, 'Position', [90, 47.2,  70, 20]);
uicontrol(groupStrain, 'Style', 'radiobutton', 'String', 'ε_xy',  'FontSize', 10, 'Position', [90, 13.6,  70, 20]);

defaultStrainBtn = findobj(groupStrain, 'String', 'ε_rot');
groupStrain.SelectedObject = defaultStrainBtn;
setappdata(fig, 'groupStrain', groupStrain);

groupNonStrain.SelectedObject = [];

%%%%%%%%%%%%%%%%%%%%%%%%%
% Height Stretch Slider %
%%%%%%%%%%%%%%%%%%%%%%%%%

sliderGroup = uipanel('Position', [0.07, 0.14, 0.21, 0.10], 'Title', 'Height Stretch Factor', 'FontSize', 10);
uicontrol(sliderGroup, 'Style', 'text', 'Position', [2, 15,  40, 20], 'String', '1',  'FontSize', 10);
uicontrol(sliderGroup, 'Style', 'text', 'Position', [160,15, 40, 20], 'String', '10', 'FontSize', 10);
stretch_slider = uicontrol(sliderGroup, 'Style', 'slider', 'Min', 1, 'Max', 10, 'Value', initial_stretch_factor, ...
                           'Position',[35,15,125,20], 'Callback', @(src,event) updateStretchFactor(src));
setappdata(fig, 'stretch_slider', stretch_slider);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Rotation Angle Slider α   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rotationPanel = uipanel('Position', [0.07, 0.02, 0.21, 0.10], 'Title', 'Rotation Angle α (°)', 'FontSize', 10);
uicontrol(rotationPanel, 'Style', 'text', 'Position', [2,  15, 40, 20], 'String', '0°',  'FontSize', 10);
uicontrol(rotationPanel, 'Style', 'text', 'Position', [160,15, 40, 20], 'String', '360°','FontSize', 10);
rotation_slider = uicontrol(rotationPanel, 'Style', 'slider', 'Min', 0, 'Max', 360, 'Value', 0, ...
                            'Tag','rotation_slider', 'Position', [35,15,125,20], ...
                            'Callback', @(src,event) updateDisplay());
setappdata(fig, 'rotation_slider', rotation_slider);

%%%%%%%%%%%%%%%%%%%%%
% Callback Functions %
%%%%%%%%%%%%%%%%%%%%%

function nonStrainChanged(event)
    fig = ancestor(event.NewValue,'figure');
    setappdata(fig,'lastChoice',event.NewValue.String);
    % deselect any in the Strain group
    gs = getappdata(fig,'groupStrain');
    gs.SelectedObject = [];
    updateDisplay();
end

function strainChanged(event)
    fig = ancestor(event.NewValue,'figure');
    setappdata(fig,'lastChoice',event.NewValue.String);
    % deselect any in the Non‐Strain group
    gns = getappdata(fig,'groupNonStrain');
    gns.SelectedObject = [];
    updateDisplay();
end

function updateColorLimitsSymmetric(src)
    newMaxValue = str2double(get(src, 'String'));
    caxis([-newMaxValue, newMaxValue]);
end

function updateColorLimitsSeparate()
    fig = gcf;
    editMin = getappdata(fig, 'editMin');
    editMax = getappdata(fig, 'editMax');
    minVal = str2double(get(editMin, 'String'));
    maxVal = str2double(get(editMax, 'String'));
    caxis([minVal, maxVal]);
end

function updateStretchFactor(src)
    stretch_factor = get(src, 'Value');
    fig = ancestor(src, 'figure');
    h_surf = getappdata(fig, 'h_surf');
    height_map = getappdata(fig, 'height_map');

    topology_flipped = flipud(height_map * stretch_factor);
    set(h_surf, 'ZData', topology_flipped);

    delete(findobj(fig, 'Type', 'patch'));
    createCube(10, stretch_factor, [0.5, 0.5, 0.5], height_map);

    drawnow;
end

function createCube(cubeSize, cubeStretchFactor, cubeColor, height_map)
    height_map_size = size(height_map);
    xBase = height_map_size(2);
    yBase = -cubeSize;
    zBase = 0;

    vertices = [xBase, yBase, zBase;
                xBase+cubeSize, yBase, zBase;
                xBase+cubeSize, yBase+cubeSize, zBase;
                xBase, yBase+cubeSize, zBase;
                xBase, yBase, zBase+(cubeSize*cubeStretchFactor);
                xBase+cubeSize, yBase, zBase+(cubeSize*cubeStretchFactor);
                xBase+cubeSize, yBase+cubeSize, zBase+(cubeSize*cubeStretchFactor);
                xBase, yBase+cubeSize, zBase+(cubeSize*cubeStretchFactor)];

    faces = [1, 2, 6, 5;
             2, 3, 7, 6;
             3, 4, 8, 7;
             4, 1, 5, 8;
             1, 2, 3, 4;
             5, 6, 7, 8];

    patch('Vertices', vertices, 'Faces', faces, 'FaceColor', cubeColor);
end

function updateDisplay()
    fig = gcf;
    h_surf = getappdata(fig, 'h_surf');
    lastChoice = getappdata(fig, 'lastChoice');
    wf = getappdata(fig,'wheelFig');
    if ~isempty(wf) && isvalid(wf) && ~strcmp(lastChoice,'Phase Map')
        close(wf);
        setappdata(fig,'wheelFig',[]);
        % restore main colorbar in case it was hidden
        cb = findobj(fig,'Tag','MainColorbar');
        if ~isempty(cb), set(cb,'Visible','on'); end
    end
    strainCorrectionToggle = getappdata(fig, 'strainCorrectionToggle');
    rotation_slider = getappdata(fig, 'rotation_slider');
    edit_pm = getappdata(fig, 'edit_pm');
    text_pm = getappdata(fig, 'text_pm');
    textMin = getappdata(fig, 'textMin');
    editMin = getappdata(fig, 'editMin');
    textMax = getappdata(fig, 'textMax');
    editMax = getappdata(fig, 'editMax');

    erot = getappdata(fig, 'erot_rippleData');
    exx = getappdata(fig, 'exx_rippleData');
    eyy = getappdata(fig, 'eyy_rippleData');
    exy = getappdata(fig, 'exy_rippleData');
    exx_corr = getappdata(fig, 'exx_rippleData_corrected');
    eyy_corr = getappdata(fig, 'eyy_rippleData_corrected');
    exy_corr = getappdata(fig, 'exy_rippleData_corrected');
    haadf = getappdata(fig, 'haadf');
    height_map = getappdata(fig, 'height_map');
    tilts = getappdata(fig, 'tilts');

    isCorrectionOn = get(strainCorrectionToggle, 'Value');
    rotation_angle = deg2rad(get(rotation_slider, 'Value'));
    
    switch lastChoice
        case 'ε_rot'
            set(text_pm, 'Visible', 'on');
            set(edit_pm, 'Visible', 'on');
            set(textMin, 'Visible', 'off');
            set(editMin, 'Visible', 'off');
            set(textMax, 'Visible', 'off');
            set(editMax, 'Visible', 'off');

            surface_color = erot;
            cbar_title = 'Rotation (°)';
            colormap(slanCM('RdBu', 256));
            flipped = flipud(surface_color);
            set(h_surf, 'CData', flipped, 'FaceColor', 'interp', 'EdgeColor', 'none');
            autoMax = max(abs(flipped(:)));
            caxis([-autoMax, autoMax]);
            set(edit_pm, 'String', num2str(autoMax));

        case 'ε_xx'
            set(text_pm, 'Visible', 'on');
            set(edit_pm, 'Visible', 'on');
            set(textMin, 'Visible', 'off');
            set(editMin, 'Visible', 'off');
            set(textMax, 'Visible', 'off');
            set(editMax, 'Visible', 'off');

            if isCorrectionOn
                [e_par, ~, ~] = rotate_strain(exx_corr, eyy_corr, exy_corr, rotation_angle);
            else
                [e_par, ~, ~] = rotate_strain(exx, eyy, exy, rotation_angle);
            end
            surface_color = e_par * 100;
            cbar_title = 'Strain (%)';
            colormap(slanCM('RdBu', 256));
            flipped = flipud(surface_color);
            set(h_surf, 'CData', flipped, 'FaceColor', 'interp', 'EdgeColor', 'none');
            autoMax = max(abs(flipped(:)));
            caxis([-autoMax, autoMax]);
            set(edit_pm, 'String', num2str(autoMax));

        case 'ε_yy'
            set(text_pm, 'Visible', 'on');
            set(edit_pm, 'Visible', 'on');
            set(textMin, 'Visible', 'off');
            set(editMin, 'Visible', 'off');
            set(textMax, 'Visible', 'off');
            set(editMax, 'Visible', 'off');

            if isCorrectionOn
                [~, e_perp, ~] = rotate_strain(exx_corr, eyy_corr, exy_corr, rotation_angle);
            else
                [~, e_perp, ~] = rotate_strain(exx, eyy, exy, rotation_angle);
            end
            surface_color = e_perp * 100;
            cbar_title = 'Strain (%)';
            colormap(slanCM('RdBu', 256));
            flipped = flipud(surface_color);
            set(h_surf, 'CData', flipped, 'FaceColor', 'interp', 'EdgeColor', 'none');
            autoMax = max(abs(flipped(:)));
            caxis([-autoMax, autoMax]);
            set(edit_pm, 'String', num2str(autoMax));

        case 'ε_xy'
            set(text_pm, 'Visible', 'on');
            set(edit_pm, 'Visible', 'on');
            set(textMin, 'Visible', 'off');
            set(editMin, 'Visible', 'off');
            set(textMax, 'Visible', 'off');
            set(editMax, 'Visible', 'off');

            if isCorrectionOn
                [~, ~, e_shear] = rotate_strain(exx_corr, eyy_corr, exy_corr, rotation_angle);
            else
                [~, ~, e_shear] = rotate_strain(exx, eyy, exy, rotation_angle);
            end
            surface_color = e_shear * 100;
            cbar_title = 'Shear Strain (%)';
            colormap(slanCM('RdBu', 256));
            flipped = flipud(surface_color);
            set(h_surf, 'CData', flipped, 'FaceColor', 'interp', 'EdgeColor', 'none');
            autoMax = max(abs(flipped(:)));
            caxis([-autoMax, autoMax]);
            set(edit_pm, 'String', num2str(autoMax));

        case 'HAADF'
            set(text_pm, 'Visible', 'off');
            set(edit_pm, 'Visible', 'off');
            set(textMin, 'Visible', 'on');
            set(editMin, 'Visible', 'on');
            set(textMax, 'Visible', 'on');
            set(editMax, 'Visible', 'on');

            surface_color = haadf;
            cbar_title = 'HAADF Intensity';
            colormap(gray);
            flipped = flipud(surface_color);
            set(h_surf, 'CData', flipped, 'FaceColor', 'interp', 'EdgeColor', 'none');
            cmin = min(flipped(:));
            cmax = max(flipped(:));
            caxis([cmin, cmax]);
            set(editMin, 'String', num2str(cmin));
            set(editMax, 'String', num2str(cmax));

        case 'Height'
            set(text_pm, 'Visible', 'off');
            set(edit_pm, 'Visible', 'off');
            set(textMin, 'Visible', 'on');
            set(editMin, 'Visible', 'on');
            set(textMax, 'Visible', 'on');
            set(editMax, 'Visible', 'on');

            surface_color = height_map;
            cbar_title = 'Height';
            colormap(hot);
            flipped = flipud(surface_color);
            set(h_surf, 'CData', flipped, 'FaceColor', 'interp', 'EdgeColor', 'none');
            cmin = min(flipped(:));
            cmax = max(flipped(:));
            caxis([cmin, cmax]);
            set(editMin, 'String', num2str(cmin));
            set(editMax, 'String', num2str(cmax));

        case 'Phase Map'
            % 1) pop up a smaller window with the color wheel
            maxTilt = getappdata(0,'maxTilt');
            fh = figure('Name','Phase Color Wheel', ...
                        'NumberTitle','off', ...
                        'Position',[1150,325,300,300]);
        
            % draw the wheel
            [xw,yw] = meshgrid(-1:2/255:1);
            mask = (xw.^2 + yw.^2) <= 1;
            phw  = atan2(yw,xw);
            ampw = sqrt(xw.^2 + yw.^2);
            ampw(~mask) = 0;
            Rw = 0.5*(sin(phw)       + 1).*ampw;
            Gw = 0.5*(sin(phw+pi/2)  + 1).*ampw;
            Bw = 0.5*(-sin(phw)      + 1).*ampw;
            Im = cat(3, Rw, Gw, Bw);
            image(Im);
            axis image off;
        
            % 2) add the text box at top
            uicontrol('Parent',fh, ...
                      'Style','text', ...
                      'String', sprintf('Maximum Tilt: %.2f°', maxTilt), ...
                      'Units','normalized', ...
                      'Position',[0.1,0.9,0.8,0.08], ...
                      'FontSize',10);
            setappdata(fig,'wheelFig',fh);    % <— store handle

            % 3) hide main GUI colorbar
            mainCB = findobj(fig,'Tag','MainColorbar');
            set(mainCB,'Visible','off');
        
            % 4) texture‐map the 3D surface (as before)
            phase = tilts(:,:,1);
            amp   = tilts(:,:,2);
            field = amp .* exp(1i*phase);
            Imf   = imag(field);
            Ref   = real(field);
            phf   = atan2(Imf,Ref);
            a     = abs(field)./max(abs(field(:)));
            Rf = 0.5*(sin(phf)      +1).*a;
            Gf = 0.5*(sin(phf+pi/2) +1).*a;
            Bf = 0.5*(-sin(phf)     +1).*a;
            Af = uint8(cat(3,Rf,Gf,Bf)*255);
            Crgb = flipud(Af);
            set(h_surf,'CData',Crgb,'FaceColor','texturemap','EdgeColor','none');
        
            return;


        otherwise
            return;
    end

    % For all but Phase Map, restore colorbar and FaceColor
    updateColorbar(cbar_title);
    set(h_surf, 'FaceColor', 'interp', 'EdgeColor', 'none');
    updateAxisRotation(rotation_angle);
    drawnow;
end

function updateColorbar(title_str)
    existing_cb = findobj(gcf, 'Tag', 'MainColorbar');
    if ~isempty(existing_cb)
        delete(existing_cb);
    end
    cb = colorbar('Tag', 'MainColorbar');
    title(cb, title_str);
    cb.FontSize   = 14;
    cb.Title.FontSize = 18;
    cb.Position   = [0.9, 0.3, 0.02, 0.4];
end

function [e_par, e_perp, e_shear] = rotate_strain(exx, eyy, exy, alpha)
    e_par   = (exx + eyy)/2 + (exx - eyy)/2 .* cos(2*alpha) + exy .* sin(2*alpha);
    e_perp  = (exx + eyy)/2 - (exx - eyy)/2 .* cos(2*alpha) - exy .* sin(2*alpha);
    e_shear = -(exx - eyy)/2 .* sin(2*alpha) + exy .* cos(2*alpha);
end

function updateAxisRotation(alpha)
    h_quiver_x = findobj(gcf, 'Type', 'Quiver', 'Color', 'r');
    h_quiver_y = findobj(gcf, 'Type', 'Quiver', 'Color', 'g');
    h_quiver_z = findobj(gcf, 'Type', 'Quiver', 'Color', 'b');
    h_text_x   = findobj(gcf, 'String', 'X');
    h_text_y   = findobj(gcf, 'String', 'Y');
    h_text_z   = findobj(gcf, 'String', 'Z');

    axis_length = 25;
    origin = [-axis_length, -axis_length, 0];

    [new_xx, new_xy] = rotate_vectors(axis_length, 0, alpha);
    [new_yx, new_yy] = rotate_vectors(0, axis_length, alpha);

    set(h_quiver_x, 'UData', new_xx, 'VData', new_xy);
    set(h_quiver_y, 'UData', new_yx, 'VData', new_yy);
    set(h_quiver_z, 'UData', 0, 'VData', 0, 'WData', axis_length);

    set(h_text_x, 'Position', origin + [new_xx, new_xy, 0]);
    set(h_text_y, 'Position', origin + [new_yx, new_yy, 0]);
    set(h_text_z, 'Position', origin + [0, 0, axis_length]);
end

function [nx_rot, ny_rot] = rotate_vectors(x, y, alpha)
    R = [cos(alpha), -sin(alpha);
         sin(alpha),  cos(alpha)];
    v = R * [x; y];
    nx_rot = v(1);
    ny_rot = v(2);
end

function updateShading(src)
    h_surf = getappdata(gcf,'h_surf');
    if get(src,'Value')  % interp = ON
        set(h_surf, ...
            'FaceColor','interp', ...
            'EdgeColor','none');
        set(src,'String','Interp On');
    else                 % interp = OFF → flat
        set(h_surf, ...
            'FaceColor','flat', ...
            'EdgeColor','none');
        set(src,'String','Interp Off');
    end
end
