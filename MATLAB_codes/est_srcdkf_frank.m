function [x,Sp]=est_srcdkf_frank(fstate,x,Sp,hmeas,z,Sq,Sr)
% UKF   Unscented Kalman Filter for nonlinear dynamic systems
% [x, P] = ukf(f,x,P,h,z,Q,R) returns state estimate, x and state covariance, P 
% for nonlinear dynamic system (for simplicity, noises are assumed as additive):
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
  [x, P] = ukf(f,x,P,h,z,Q,R);            % ekf 
  xV(:,k) = x;                            % save estimate
  s = f(s) + q*randn(3,1);                % update process 
end
for k=1:3                                 % plot results
  subplot(3,1,k)
  plot(1:N, sV(k,:), '-', 1:N, xV(k,:), '--')
end
%}
% Reference: Julier, SJ. and Uhlmann, J.K., Unscented Filtering and
% Nonlinear Estimation, Proceedings of the IEEE, Vol. 92, No. 3,
% pp.401-422, 2004. 
%
% By Yi Cao at Cranfield University, 04/01/2008
%
L=numel(x);                                 %numer of states
m=numel(z);                                 %numer of measurements
h=sqrt(3);                                  %scaling factor

Wm = [(h^2 - L)/(h^2)     (1/(2*(h^2))+zeros(1,2*L))];               % weights for means 
Wc1_qr =[(1/(2 * h))+zeros(1,L)];           % weights for covariance
Wc2_qr =[(sqrt(h^2-1)/(2*(h^2)))+zeros(1,L)];       % weights for covariance

X=sigmas(x,Sp,h);                            %sigma points around x
[x1,X1,Sp1,X2]=cdt1(fstate,X,Wm,Wc1_qr, Wc2_qr,L,Sq);          %unscented transformation of process
X1=sigmas(x1,Sp1,h);                          %sigma points around x1
% X2=X1-x1(:,ones(1,size(X1,2)));             %deviation of X1
[z1,Z1,Sp2,Z2]=cdt2(hmeas,X1,Wm,Wc1_qr, Wc2_qr,m,Sr);       %unscented transformation of measurments
Wc_1=(Wc1_qr(1))*eye(L);
P12=Wc_1*Sp1'*(Z1(:, 2:L+1)-Z1(:, L+2:2*L+1))';
%P12=X2*diag(Wc)*Z2';                        %transformed cross-covariance
K=(P12 / Sp2) / Sp2';
x=x1+K*(z-z1);                              %state update
% P=P1-K*P12';                                %covariance update
U = K*Sp2';
for i = 1:m
    Sp1 = cholupdate(Sp1, U(:,i), '-');
    %Sp1 = Sp1';
end
Sp=Sp1;

%Sp=chol(Sp1'*Sp1 - (P12 / Sp2)*(P12 / Sp2)');



function [y,Y,Sp,Ym]=cdt1(f,X,Wm,Wc1,Wc2,n,Sq)
%Unscented Transformation
%Input:
%        f: nonlinear map
%        X: sigma points
%       Wm: weights for mean
%       Wc: weights for covraiance
%        n: numer of outputs of f
%        R: additive covariance
%Output:
%        y: transformed mean
%        Y: transformed smapling points
%        P: transformed covariance
%       Y1: transformed deviations
L=size(X,2);
y=zeros(n,1);
Y=zeros(n,L);
for k=1:L                   
    Y(:,k)=f(X(:,k));       
    y=y+Wm(k)*Y(:,k);       
end
Ym=Y-y(:,ones(1,L));
%Y_inter=y(:, 1).*log(Y);
%Y1=Y_inter(:, ones(1, L));
% P=Y1*diag(Wc)*Y1'+Q;
%P=Y1*diag(Wc)*Y1';
Y1=Y(:,2:(n+1))-Y(:,n+2:2*n+1);
P1=Y1*diag(abs(Wc1));
Y2=Y(:,2:(n+1))+Y(:,n+2:2*n+1)-2*repmat(Y(:,1),1,n);
P2=Y2*diag(abs(Wc2));
%Sp = P1+P2 + Sq;
%P = P1+P2 +Q;
[~, R] = qr([P1 P2 Sq]', 0); 
Sp = R;



function [y,Y,Sp,Ym]=cdt2(f,X,Wm,Wc1,Wc2,n,Sr)
%Unscented Transformation
%Input:
%        f: nonlinear map
%        X: sigma points
%       Wm: weights for mean
%       Wc: weights for covraiance
%        n: numer of outputs of f
%        R: additive covariance
%Output:
%        y: transformed mean
%        Y: transformed smapling points
%        P: transformed covariance
%       Y1: transformed deviations

L=size(X,2);
m = (L-1)/2;
y=zeros(n,1);
Y=zeros(n,L);
for k=1:L                   
    Y(:,k)=f(X(:,k));       
    y=y+Wm(k)*Y(:,k);       
end
Ym=Y-y(:,ones(1,L));
%Y_inter=y(:, 1).*log(Y);
%Y1=Y_inter(:, ones(1, L));
% Y1=Y(:,2:(m+1))-Y(:,m+2:2*m+1);
% P1=Y1*diag(Wc1)*Y1';
% Y2=Y(:,2:(m+1))+Y(:,m+2:2*m+1)-2*repmat(Y(:,1),1,m);
% P2=Y2*diag(Wc2)*Y2';
% Sp = P1+P2 +Sr;

Y1=Y(:,2:(m+1))-Y(:,m+2:2*m+1);
diag_Wc1=diag(Wc1);
P1=Y1*diag_Wc1;
Y2=Y(:,2:(m+1))+Y(:,m+2:2*m+1)-2*repmat(Y(:,1),1,m);
diag_Wc2=diag(Wc2);
P2=Y2*diag_Wc2;
%Sp = P1+P2 + Sq;
%P = P1+P2 +Q;
[~, R] = qr([P1 P2 Sr]', 0); 
Sp = R;



function X=sigmas(x,Sp,h)

%Sigma points around reference point
%Inputs:
%       x: reference point
%       P: covariance
%       c: coefficient
%Output:
%       X: Sigma points
%A = h*sqrtm(P)';
A = h*Sp';
% A=c*P^0.5;
Y = x(:,ones(1,numel(x)));
X = [x Y+A Y-A]; 
% A = c*cholcov(P)';
% % A=c*P^0.5;
% % A=real((n*P)^0.5);
% n=numel(x);
% [A_rows,A_cols]=size(A);
% if A_rows==0,
%     Y = x(:,ones(1,n));
%     X = [x Y Y] ;
% else
%     A=[zeros(A_rows,n-A_cols) A];
%     Y = x(:,ones(1,n));
%     X = [x Y+A Y-A] ;
% end