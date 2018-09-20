function [x, P] = ckf_frank_v1(f_state, xhat, Pplus, h_meas, z, Q, R)
%global Q R fai gama kesi w m;
%% ---------------------------3rd-degree CKF----------------------

n=numel(xhat);   %numer of states
%n = 4; %dimension of the system
m = 2 * n; % Number of cubature points

% 3rd-degree CKF/SCKF
w = 1 / m; % Weight
kesi1 = eye(n);
kesi2 = -eye(n);
kesi = [kesi1, kesi2] * (sqrt(n)); % Construction of the cubature points

%% -----------------------------Time Update-----------------------
% Evaluate the Cholesky factor
%Pplus=topdm(Pplus);
%Pplus=qr(Pplus);
Shat = sqrtm(Pplus);
%Shat = chol(Pplus);
%Shat = qr(Pplus);
for cpoint = 1 : m
    % Evaluate the cubature points
    rjpoint(:, cpoint) = Shat * kesi(:, cpoint) + xhat;
    % Evaluate the propagated cubature points
    %Xminus(:, cpoint) = fai * rjpoint(:, cpoint);
    Xminus(:, cpoint) =  f_state(rjpoint(:, cpoint));
end

% Estimate the predicted state
xminus = w * sum(Xminus, 2); %sum(A,2) is a column vector containing the sum of each row.

% Estimate the predicted error covariance
%Pminus = w * (Xminus * Xminus') - xminus * xminus' + gama * Q * gama';
%Q2 = diag([10^-700*ones(1,4) 10^-210 10^-200]);
%Pminus = w * (Xminus * Xminus') - xminus * xminus' + Q2;

Pminus = w * (Xminus * Xminus') - xminus * xminus' + Q;
%% ---------------------------------------------------------------

%% -------------------------Measurement Update--------------------
% Evaluate the Cholesky factor
%Pminus=topdm(Pminus);
Sminus = sqrtm(Pminus);
%Sminus = chol(Pminus);
%Sminus = qr(Pminus);
for cpoint = 1 : m
    % Evaluate the cubature points
    rjpoint1(:, cpoint) = Sminus * kesi(:, cpoint) + xminus;
    % Evaluate the propagated cubature points
    %Z(cpoint) = atan(rjpoint1(3, cpoint) / rjpoint1(1, cpoint));
    %L=size(rjpoint1,2);
    %y=zeros(n,1);
    Z(:, cpoint)=h_meas(rjpoint1(:, cpoint));
    %rjpoint1_h=[rjpoint1; ones(2, size(rjpoint1, 2))*0];
    %Z(:, cpoint)=h_meas(rjpoint1_h(:, cpoint));
end
% Estimate the predicted measurement
zhat = w * sum(Z, 2);

% Estimate the innovation covariance matrix
%Pzminus = w * sum(Z * Z') - zhat^2 + R; %????
%Pzminus = w * sum(Z * Z') - zhat^2 + R;
Pzminus = w * Z * Z' - zhat*(zhat') + R;
%Pzminus = w * sum(Z * Z') - zhat*(zhat');

% Estimate the cross-covariance matrix
Pxzminus = w * rjpoint1 * Z' - xminus * zhat';

% Estimate the Kalman gain
W = Pxzminus * inv(Pzminus);

% Estimate the updated state
x = xminus + W * (z - zhat);

% Estimate the correspondong error covariance
P = Pminus - W * Pzminus * W';
%P = -P;