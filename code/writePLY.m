function writePLY(varargin)
% WRITEPLY - Export mesh to PLY format and visualize
%
% SYNTAX:
%   writePLY(vertices, faces, filename)
%   writePLY(vertices, faces, filename, 'binary', true)
%   writePLY(TR, filename)                                   % Triangulation object
%   writePLY(TR, filename, 'binary', true)                   % Triangulation object
%
% INPUTS:
%   vertices - N x 3 matrix of vertex coordinates [x, y, z]
%   faces    - M x 3 matrix of face vertex indices (1-based)
%   TR       - triangulation object (alternative to vertices/faces)
%   filename - output PLY filename (string or char array)
%
% OPTIONS:
%   'binary' - logical, true for binary PLY, false for ASCII (default: false)
%
% OUTPUTS:
%   - Creates PLY file compatible with Python trimesh, open3d, meshlab
%   - Displays mesh visualization (solid faces + wireframe views)
%
% EXAMPLES:
%   writePLY(vertices, faces, 'mesh.ply')                    % ASCII format
%   writePLY(vertices, faces, 'mesh.ply', 'binary', true)    % Binary format
%   writePLY(TR, 'mesh.ply')                                 % From triangulation
%   writePLY(TR, 'mesh.ply', 'binary', true)                 % Binary triangulation
%
% REQUIREMENTS:
%   - No additional toolboxes required
%   - Uses only built-in MATLAB functions
%
% NOTES:
%   - Function converts from MATLAB 1-based to PLY 0-based indexing
%   - Handles NaN/Inf values in vertices
%   - Creates visualization before export for verification

% Determine input format and parse arguments
if nargin >= 1 && isa(varargin{1}, 'triangulation')
    % Format: writePLY(TR, filename, [options])
    TR = varargin{1};
    vertices = TR.Points;
    faces = TR.ConnectivityList;
    
    p = inputParser;
    addRequired(p, 'TR', @(x) isa(x, 'triangulation'));
    addRequired(p, 'filename', @(x) ischar(x) || isstring(x));
    addParameter(p, 'binary', false, @islogical);
    
    parse(p, varargin{:});
    filename = char(p.Results.filename);
    binary_format = p.Results.binary;
    
elseif nargin >= 3 && isnumeric(varargin{1}) && isnumeric(varargin{2})
    % Format: writePLY(vertices, faces, filename, [options])
    p = inputParser;
    addRequired(p, 'vertices', @(x) isnumeric(x) && size(x,2)==3);
    addRequired(p, 'faces', @(x) isnumeric(x) && size(x,2)==3);
    addRequired(p, 'filename', @(x) ischar(x) || isstring(x));
    addParameter(p, 'binary', false, @islogical);
    
    parse(p, varargin{:});
    vertices = p.Results.vertices;
    faces = p.Results.faces;
    filename = char(p.Results.filename);
    binary_format = p.Results.binary;
    
else
    error('writePLY:InvalidInput', [...
        'Invalid input format. Use either:\n' ...
        '  writePLY(vertices, faces, filename, options...)\n' ...
        '  writePLY(TR, filename, options...)']);
end

%% Input validation
if size(vertices, 2) ~= 3
    error('writePLY:InvalidVertices', 'Vertices must be N x 3 matrix [x, y, z]');
end
if size(faces, 2) ~= 3
    error('writePLY:InvalidFaces', 'Faces must be M x 3 matrix of triangular faces');
end

num_vertices = size(vertices, 1);
num_faces = size(faces, 1);

% Validate face indices
if any(faces(:) < 1) || any(faces(:) > num_vertices)
    error('writePLY:InvalidIndices', 'Face indices must be between 1 and %d', num_vertices);
end

% Check for and handle NaN/Inf values
bad_vertices = any(~isfinite(vertices), 2);
if any(bad_vertices)
    warning('writePLY:BadVertices', '%d vertices contain NaN/Inf values - removing them', sum(bad_vertices));
    % Remove bad vertices and update face indices accordingly
    good_vertices = ~bad_vertices;
    vertices = vertices(good_vertices, :);
    
    % Create mapping from old to new vertex indices
    vertex_map = zeros(num_vertices, 1);
    vertex_map(good_vertices) = 1:sum(good_vertices);
    
    % Update faces and remove faces with bad vertices
    faces_updated = vertex_map(faces);
    valid_faces = all(faces_updated > 0, 2);
    faces = faces_updated(valid_faces, :);
    
    num_vertices = size(vertices, 1);
    num_faces = size(faces, 1);
end

%% Create visualization
figure('Name', sprintf('PLY Export: %s', filename), 'Position', [100, 100, 800, 600]);

% Combined solid surface with wireframe overlay
trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3), ...
        'FaceColor', [0.7 0.8 0.9], ...
        'EdgeColor', [0.2 0.4 0.6], ...
        'LineWidth', 0.5, ...
        'FaceAlpha', 0.95);

axis equal; axis tight; grid on;
xlabel('X'); ylabel('Y'); zlabel('Z');
title(sprintf('Mesh Visualization: %s', filename), 'Interpreter', 'none');
view(3);
camlight('headlight'); 
lighting gouraud;

% Add mesh statistics as text
stats_text = sprintf(['Mesh Statistics:\n' ...
                     'Vertices: %d\n' ...
                     'Faces: %d\n' ...
                     'Format: %s\n' ...
                     'Export: %s'], ...
                     num_vertices, num_faces, ...
                     ternary(binary_format, 'Binary', 'ASCII'), ...
                     filename);

% Add text box with statistics (positioned for single view)
annotation('textbox', [0.02, 0.02, 0.35, 0.20], ...
          'String', stats_text, ...
          'FontSize', 9, ...
          'BackgroundColor', 'white', ...
          'EdgeColor', 'black');

% Display mesh info in command window
fprintf('\n=== PLY EXPORT ===\n');
fprintf('File: %s\n', filename);
fprintf('Format: %s\n', ternary(binary_format, 'Binary', 'ASCII'));
fprintf('Vertices: %d\n', num_vertices);
fprintf('Faces: %d\n', num_faces);
fprintf('Bounds: X[%.3f,%.3f] Y[%.3f,%.3f] Z[%.3f,%.3f]\n', ...
        min(vertices(:,1)), max(vertices(:,1)), ...
        min(vertices(:,2)), max(vertices(:,2)), ...
        min(vertices(:,3)), max(vertices(:,3)));

%% Export PLY file
if binary_format
    write_binary_ply(filename, vertices, faces);
else
    write_ascii_ply(filename, vertices, faces);
end

% Verify export success
file_info = dir(filename);
if isempty(file_info)
    error('writePLY:ExportFailed', 'Failed to create PLY file: %s', filename);
else
    fprintf('✓ Export successful: %.1f KB\n', file_info.bytes / 1024);
end

end

%% ASCII PLY writer (default)
function write_ascii_ply(filename, vertices, faces)
    num_vertices = size(vertices, 1);
    num_faces = size(faces, 1);
    
    fid = fopen(filename, 'w');
    if fid == -1
        error('writePLY:FileError', 'Cannot create file: %s', filename);
    end
    
    try
        % Write ASCII PLY header
        fprintf(fid, 'ply\n');
        fprintf(fid, 'format ascii 1.0\n');
        fprintf(fid, 'comment MATLAB writePLY function export\n');
        fprintf(fid, 'element vertex %d\n', num_vertices);
        fprintf(fid, 'property float x\n');
        fprintf(fid, 'property float y\n');
        fprintf(fid, 'property float z\n');
        fprintf(fid, 'element face %d\n', num_faces);
        fprintf(fid, 'property list uchar int vertex_indices\n');
        fprintf(fid, 'end_header\n');
        
        % Write vertices (high precision for accuracy)
        fprintf(fid, '%.8f %.8f %.8f\n', vertices');
        
        % Write faces (convert from 1-based to 0-based indexing)
        faces_0based = faces - 1;
        fprintf(fid, '3 %d %d %d\n', faces_0based');
        
        fclose(fid);
        
    catch ME
        fclose(fid);
        if exist(filename, 'file')
            delete(filename);
        end
        rethrow(ME);
    end
end

%% Binary PLY writer (for large meshes)
function write_binary_ply(filename, vertices, faces)
    num_vertices = size(vertices, 1);
    num_faces = size(faces, 1);
    
    % Write header in text mode
    fid = fopen(filename, 'w');
    if fid == -1
        error('writePLY:FileError', 'Cannot create file: %s', filename);
    end
    
    try
        % Write binary PLY header
        fprintf(fid, 'ply\n');
        fprintf(fid, 'format binary_little_endian 1.0\n');
        fprintf(fid, 'comment MATLAB writePLY function binary export\n');
        fprintf(fid, 'element vertex %d\n', num_vertices);
        fprintf(fid, 'property float x\n');
        fprintf(fid, 'property float y\n');
        fprintf(fid, 'property float z\n');
        fprintf(fid, 'element face %d\n', num_faces);
        fprintf(fid, 'property list uchar int vertex_indices\n');
        fprintf(fid, 'end_header\n');
        fclose(fid);
        
        % Append binary data
        fid = fopen(filename, 'a');
        
        % Write vertices as binary float32
        fwrite(fid, vertices', 'float32');
        
        % Write faces as binary
        faces_0based = int32(faces - 1);
        for i = 1:num_faces
            fwrite(fid, 3, 'uint8');  % 3 vertices per face
            fwrite(fid, faces_0based(i, :), 'int32');
        end
        
        fclose(fid);
        
    catch ME
        if fid ~= -1
            fclose(fid);
        end
        if exist(filename, 'file')
            delete(filename);
        end
        rethrow(ME);
    end
end

%% Utility function for conditional text
function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end