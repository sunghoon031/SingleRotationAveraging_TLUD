close all; clear all; clc;

%% Simulate rotations

n_rotations = 1000; %[3:3:99, 110:10:500];%[3:3:99, 110:10:300, 350:50:1000];%[10:10:100, 200:100:1000];
angle_deg_std = 5;
outlier_ratio = 0.9;

R_true = RandomRotationMatrix;

R_samples = cell(1, n_rotations);
for i = 1:n_rotations
     if (i <= round(outlier_ratio*n_rotations))
        R_samples{i} = RandomRotationMatrix;
     else
        R_samples{i} = NoiseRotationMatrix(angle_deg_std)*R_true;
     end

end

%% Perform rotation averaging

n_steps = 10;
epsilon_c = 0.5;
delta_thr = 0.001; 
R = AverageRotations(R_samples, n_steps, epsilon_c, delta_thr);

error_in_deg = abs(acosd((trace(R_true*R')-1)/2))              


          




%% Function definitions

function R_out = AverageRotations(R_samples, nSteps, epsilon_c, delta_thr)

    % 1. Find inliers:
    
    n_samples = length(R_samples);
        
    vectors_total = zeros(9,n_samples);
    for i = 1:n_samples
        vectors_total(:,i)= R_samples{i}(:);
    end  
    
    min_cost = inf;
    for i = 1:n_samples
        vec_i = vectors_total(:,i);
        
        vec_diff = vectors_total-vec_i;
        vec_diff = vec_diff.^2;
        es = sqrt(sum(vec_diff,1));
        
        
        es(es>epsilon_c) = epsilon_c;
        cost = sum(es);
        
        if (cost < min_cost)
            min_cost = cost;
            inliers = find(es<epsilon_c);
        end
    
    end

    % 2. Initialize the rotation using the inlier set:
    
    R_sum = zeros(3,3);
    for i = inliers
        R_sum = R_sum + R_samples{i};
    end
    [U,~,V] = svd(R_sum);
    R_out = U*V.';
    if (det(R_out) < 0)
        V(:,3) = -V(:,3);
        R_out = U*V.';
    end
    
    % 3. Find the L-1 mean of the inliers:
    
    n_inliers = length(inliers);

    q_out = R2q(R_out);
    
    q_all = zeros(4,n_inliers);
    for i = 1:n_inliers
        q_all(:,i) = R2q(R_samples{inliers(i)});
    end


    q_all_inv_q_out = zeros(4, n_inliers);

    for j = 1:nSteps

        q_all_inv_q_out(1,:) = ... %scalar term of (q_all)*inv(q_out)
            -q_all(1,:).*q_out(1,:) - sum(q_all(2:4,:).*q_out(2:4,:),1); 

        q_all_inv_q_out(2:4,:) = ... % vector term of (q_all)*inv(q_out)
           +q_all(1,:).*q_out(2:4,:) - q_out(1,:).*q_all(2:4,:) ...
           + [q_all(3,:).*q_out(4,:) - q_all(4,:).*q_out(3,:);...
              q_all(4,:).*q_out(2,:) - q_all(2,:).*q_out(4,:);...
              q_all(2,:).*q_out(3,:) - q_all(3,:).*q_out(2,:)];


        sine_half = sqrt(sum(q_all_inv_q_out(2:4,:).^2, 1));
        theta = 2*atan2(sine_half, q_all_inv_q_out(1,:));
        theta(theta < -pi) = theta(theta < -pi) + 2*pi;
        theta(theta > pi) = theta(theta > pi) - 2*pi;

        q_all_inv_q_out(2:4,:) = q_all_inv_q_out(2:4,:).*sign(theta);
        theta = abs(theta);

        unit_v = q_all_inv_q_out(2:4,:)./sine_half;

        delta = sum(unit_v, 2)/sum(1./theta);
        delta_angle = norm(delta);

        unit_delta = delta/delta_angle;

        q_delta = zeros(4,1);
        q_delta(1) = cos(delta_angle/2);
        q_delta(2:4) = unit_delta*sin(delta_angle/2);

        q_out_ = q_out;
        q_out_(1,:) = ... %scalar term of (q_delta)*q_geo1
            q_delta(1).*q_out(1) - sum(q_delta(2:4).*q_out(2:4),1); 

        q_out_(2:4,:) = ... % vector term of (q_delta)*q_geo1
           q_delta(1).*q_out(2:4) + q_out(1,:).*q_delta(2:4) ...
           + [q_delta(3).*q_out(4) - q_delta(4,:).*q_out(3);...
              q_delta(4).*q_out(2) - q_delta(2,:).*q_out(4);...
              q_delta(2).*q_out(3) - q_delta(3,:).*q_out(2)];

        q_out = q_out_;  

        if (delta_angle < delta_thr)
            break;
        end
    end

    R_out = q2R(q_out);

end


function R = q2R(q)
    qw = q(1); qx = q(2); qy = q(3); qz = q(4);
    R = zeros(3,3);
    R(1,1) = 1 - 2*qy^2 - 2*qz^2;
    R(1,2) = 2*qx*qy - 2*qz*qw;
    R(1,3) = 2*qx*qz + 2*qy*qw;
    R(2,1) = 2*qx*qy + 2*qz*qw;
    R(2,2) = 1 - 2*qx^2 - 2*qz^2;
    R(2,3) = 2*qy*qz - 2*qx*qw;
    R(3,1) = 2*qx*qz - 2*qy*qw;
    R(3,2) = 2*qy*qz + 2*qx*qw;
    R(3,3) = 1 - 2*qx^2 - 2*qy^2;
end

function q = R2q(R)
    % https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    q = zeros(4,1);
    tr = trace(R);
    if (tr > 0)
        s = sqrt(1+tr)*2; %s = 4*qw
        q(1) = 0.25*s;
        q(2) = (R(3,2)-R(2,3))/s;
        q(3) = (R(1,3)-R(3,1))/s;
        q(4) = (R(2,1)-R(1,2))/s;
    elseif (R(1,1) > R(2,2) && R(1,1) > R(3,3))
        s = sqrt(1+R(1,1)-R(2,2)-R(3,3))*2; %s = 4*qx
        q(1) = (R(3,2)-R(2,3))/s;
        q(2) = 0.25*s;
        q(3) = (R(1,2)+R(2,1))/s;
        q(4) = (R(1,3)+R(3,1))/s;
    elseif (R(2,2) > R(3,3))
        s = sqrt(1+R(2,2)-R(1,1)-R(3,3))*2; %s = 4*qy
        q(1) = (R(1,3)-R(3,1))/s;
        q(2) = (R(1,2)+R(2,1))/s;
        q(3) = 0.25*s;
        q(4) = (R(2,3)+R(3,2))/s;
    else
        s = sqrt(1+R(3,3)-R(1,1)-R(2,2))*2; %s = 4*qz
        q(1) = (R(2,1)-R(1,2))/s;
        q(2) = (R(1,3)+R(3,1))/s;
        q(3) = (R(2,3)+R(3,2))/s;
        q(4) = 0.25*s;
    end
end


function [R] = NoiseRotationMatrix(angle_deg_std)
    axis_perturb = rand(3,1)-0.5;
    axis_perturb = axis_perturb/norm(axis_perturb);
    angle_rad_std = angle_deg_std/180*pi;
    angle_perturb = normrnd(0,angle_rad_std);
    
    ex = [0 0 0; 0 0 -1; 0 1 0];
    ey = [0 0 1; 0 0 0; -1 0 0];
    ez = [0 -1 0; 1 0 0; 0 0 0];
    ss_mat = axis_perturb(1)*ex+axis_perturb(2)*ey+axis_perturb(3)*ez;
    
    R = eye(3)+ss_mat*sin(angle_perturb)+ss_mat^2*(1-cos(angle_perturb));

end

function [R] = RandomRotationMatrix()
    
    % Each column of a rotation matrix can be thought of as the x, y and
    % z-axis of some reference frame placed with some orientation.
    
    % We set x-axis to be a random 3D unit vector.
    % For it to have a "random" direction, we generate a random point
    % inside a unit sphere and use its direction.
    
    while (true)
        p = 2*(rand(3,1)-0.5);
        if (sum(p.*p) < 1)
            break;
        end
    end
    x_axis = p/norm(p);
    
    % Then, we generate y-axis by finding a vector that is perpendicular to
    % the x-axis. This is done by generating another random 3D unit vector
    % and then computing the cross product with the x-axis.
    
    while (true)
        p = 2*(rand(3,1)-0.5);
        if (sum(p.*p) < 1)
            break;
        end
    end
    
    y_axis = cross(x_axis, p);
    y_axis = y_axis/norm(y_axis);
    
    % z-axis must be the cross product of x-axis and y-axis.
    z_axis = cross(x_axis, y_axis);
    z_axis = z_axis/norm(z_axis);
    
    R = [x_axis, y_axis, z_axis];


end

