function [x, y, scores, Ih, Iv] = extract_keypoints(image)
    % Set k to 0.05
    k = 0.05;
    
    % Set window size to 5
    window_size = 5;
    
    % Calculate half_window_size. Since we are looking for all neighbors
    % less than half, we can subtract 1 from 5 and then take half to get
    % the floor of the division.
    half_window_size = (window_size-1)/2;
    
    % Convert image to grayscale
    im = rgb2gray(imread(image));
    
    % Convert image to double
    im = double(im);
    
    % Define a filter to calculate horizontal and vertical gradients
    sobel = fspecial('sobel');
    
    % Compute the horizontal gradient
    Ih = imfilter(im,sobel);
    
    % Compute the vertical gradient
    Iv = imfilter(im,sobel');
    
    % Cast Ih and Iv to double
    Ih = double(Ih);
    Iv = double(Iv);
    
    % Get image dimensions
    rows = size(im,1);
    cols = size(im,2);
    
    % Initialize matrix R to the same size as the original image
    R = zeros(rows,cols);
    
    for i=1:rows
        for j=1:cols
            % If one of the i or j values is out of bounds, set R at (i,j)
            % to -inf
            if ((i-half_window_size <= 0) || (i+half_window_size > rows)...
                    || (j-half_window_size <= 0) || (j+half_window_size ...
                    > cols))
                R(i,j) = -Inf;
            else
                % Initialize 2x2 matrix M
                M = zeros(2);
                for r=i-half_window_size:i+half_window_size
                    for c=j-half_window_size:j+half_window_size
                        M(1,1) = M(1,1) + Ih(r,c)^2; 
                        M(1,2) = M(1,2) + (Ih(r,c) * Iv(r,c));
                        M(2,1) = M(2,1) + (Ih(r,c) * Iv(r,c));
                        M(2,2) = M(2,2) + Iv(r,c)^2;
                    end
                end
                R(i,j) = det(M) - (k * (trace(M)))^2;
            end
        end
    end
    
    % Create a vector from the R Matrix
    R_vector = R(:);
    
    % Find all the indexes where -inf are
    inf_index = find(isinf(R_vector));
    
    % Set the -inf values in R_vector to 0
    R_vector(inf_index) = [];
    
    % Find the average of R to threshold results. Use R_vector, which has
    % no -inf, to calculate mean
    R_mean = mean(R_vector);
    
    % All values in R that are less than 5 times the mean are set to 0
    R(R < (R_mean*5)) = 0;
    
    % Initialize vectors for x, y, and scores
    x = [];
    y = [];
    scores = [];
    
    for i=1:rows
        for j=1:cols
            % If one of the i or j values is out of bounds/has no
            % neighbors, continue
            if ((i-half_window_size <= 0) || (i+half_window_size > rows) || (j-half_window_size <= 0) || (j+half_window_size > cols))
                continue;
            % If R(i,j) is greater than all of its neighbors, add the R
            % score to scores, the i value to y, and the j value to x
            elseif (R(i,j) > R(i-half_window_size,j-half_window_size) && ...
                    R(i,j) > R(i,j-half_window_size) && ...
                    R(i,j) > R(i+half_window_size,j-half_window_size) && ...
                    R(i,j) > R(i-half_window_size,j) && ...
                    R(i,j) > R(i+half_window_size,j) && ...
                    R(i,j) > R(i-half_window_size,j+half_window_size) && ...
                    R(i,j) > R(i,j+half_window_size) && ...
                    R(i,j) > R(i+half_window_size,j+half_window_size))
                scores = [scores R(i,j)];
                x = [x j];
                y = [y i];
            end
        end
    end
    
    figure;
    imshow(image);
    hold on;
    for i=1:length(x)
        plot(x(i), y(i), 'ro', 'MarkerSize', scores(i) / 100000000000);
    end
    hold off;
end

