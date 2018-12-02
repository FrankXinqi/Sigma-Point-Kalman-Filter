function [x,P]=ekf_samson(fstate,x,P,hmeas,z,Q,R,lnr_A,lnr_H,V_kminus1,theta_kminus1,V_k,theta_k,vdr,vqr,vdg,vqg)
% EKF   Extended Kalman Filter for nonlinear dynamic systems
% [x, P] = ekf(f,x,P,h,z,Q,R) returns state estimate, x and state covariance, P 
% for nonlinear dynamic system:
%           x_k+1 = f(x_k) + w_k
%           z_k   = h(x_k) + v_k
% where w ~ N(0,Q) meaning w is gaussian noise with covariance Q
%       v ~ N(0,R) meaning v is gaussian noise with covariance R
% Inputs:   f: function handle for f(x)
%           x: "a priori" state estimate
%           P: "a priori" estimated state covariance
%           h: function handle for h(x)
%           z: current measurement
%           Q: process noise covariance 
%           R: measurement noise covariance
% Output:   x: "a posteriori" state estimate
%           P: "a posteriori" state covariance
%
% Example:
%{
n=3;      %number of state
q=0.1;    %std of process 
r=0.1;    %std of measurement
Q=q^2*eye(n); % covariance of process
R=r^2;        % covariance of measurement  
f=@(x)[x(2);x(3);0.05*x(1)*(x(2)+x(3))];  % nonlinear state equations
h=@(x)x(1);                               % measurement equation
s=[0;0;1];                                % initial state
x=s+q*randn(3,1); %initial state          % initial state with noise
P = eye(n);                               % initial state covraiance
N=20;                                     % total dynamic steps
xV = zeros(n,N);          %estmate        % allocate memory
sV = zeros(n,N);          %actual
zV = zeros(1,N);
for k=1:N
  z = h(s) + r*randn;                     % measurments
  sV(:,k)= s;                             % save actual state
  zV(k)  = z;                             % save measurment
  [x, P] = ekf(f,x,P,h,z,Q,R);            % ekf 
  xV(:,k) = x;                            % save estimate
  s = f(s) + q*randn(3,1);                % update process 
end
for k=1:3                                 % plot results
  subplot(3,1,k)
  plot(1:N, sV(k,:), '-', 1:N, xV(k,:), '--')
end
%}
% By Yi Cao at Cranfield University, 02/01/2008
%
%% begin
x1=fstate(x);   % nonlinear update
A=lnr_A(x,V_kminus1,theta_kminus1,vdr,vqr,vdg,vqg);  % linearized matrix A
P=A*P*A'+Q;              %partial update.
% P=A*P*A;
z1=hmeas(x1);
H=lnr_H(x1,V_k,theta_k,vdr,vqr,vdg,vqg); % linearized matrix B
% P12=P*H';                   %cross covariance
% K=P12*inv(H*P12+R);       %Kalman filter gain
% x=x1+K*(z-z1);            %state estimate
% P=P-K*P12';               %state covariance matrix
% S=chol(H*P*H'+R);            %Cholesky factorization. R is upper trangular, R'R=A 
% S=chol(H*P12+R);            %Cholesky factorization
% U=P12/S;                    %K=U/R'; Faster because of back substitution
% x=x1+U*(S'\(z-z1));         %Back substitution to get state update
% P=P-U*U';                   %Covariance update, U*U'=P12/R/R'*P12'=K*P12.

S=H*P*H'+R;
K=P*H'/S;                  %Near-optimal Kalman gain;K=U/R'; Faster because of back substitution
x=x1+K*(z-z1);         %Back substitution to get state update
P=P-K*H*P;                   %Covariance update, U*U'=P12/R/R'*P12'=K*P12.

% function [z,A]=jaccsd(fun,x)
% % JACCSD Jacobian through complex step differentiation
% % [z J] = jaccsd(f,x)
% % z = f(x)
% % J = f'(x)
% %
% z=fun(x);
% n=numel(x);
% m=numel(z);
% A=zeros(m,n);
% h=n*eps;
% for k=1:n
%     x1=x;
%     x1(k)=x1(k)+1i*h;
%     A(:,k)=imag(fun(x1))/h;
% end