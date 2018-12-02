function [x,P] = est_ukf_kia(fstate,x,P,hmeas,z,Q,R)
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
alpha=3;
m_q=alpha^L;
m=numel(z);                                 %numer of measurements

J=0*eye(m_q);
for i=1:(m_q-1)
    J(i, i+1)=sqrt(i/2);
    J(i+1, i)=sqrt(i/2);
end

[V,D] = eig(J);

kesi = sqrt(2)*D;

for i=1:m_q
    v(i) = V(729, i); 
    w(i) = v(i)^2;
end

X=sigmas(x,P,kesi);                            %sigma points around x
[x1,X1,SP1,X2]=ut1(fstate,X,w,w,L,Q);          %unscented transformation of process
%X1=sigmas(x1,P1,kesi);                         %sigma points around x1
% X2=X1-x1(:,ones(1,size(X1,2)));             %deviation of X1
[z1,Z1,SP2,Z2]=ut2(hmeas,X1,w,w,m,R);       %unscented transformation of measurments
%P12=X1*diag(w)*Z1'-x1*z1';                        %transformed cross-covariance
P12=X2*Z2';
K=P12/SP2'/SP2;
x=x1+K*(z-z1);                              %state update
%P=P1-K*P2*K';                                %covariance update
%P=qr([X2-K*Z2 K*R]);

U = K*SP2';
for i = 1:m
    SP1 = cholupdate(SP1, U(:,i), '-');
end
P=SP1;


function [y,Y,Sp,Y1]=ut1(f,X,Wm,Wc,n,SQ)
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
Y1=Y-y(:,ones(1,L));
%Y_inter=y(:, 1).*log(Y);
%Y1=Y_inter(:, ones(1, L));

for i=1:L
    sqrt_w(i)=sqrt(Wc(i));
end
yy = repmat(y, 1, L);
Y1=( Y - yy )*diag(sqrt_w);
[~, R] = qr([ ( Y - yy )*diag(sqrt_w), SQ]', 0);

Sp = R;

%P=Y*diag(Wc)*Y'-y*y'+Q;
%P=Y1*diag(Wc)*Y1';

function [y,Y,Sp,Y1]=ut2(f,X,Wm,Wc,n,SR)
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

%Y_inter=y(:, 1).*log(Y);
%Y1=Y_inter(:, ones(1, L));
%P=Y*diag(Wc)*Y'-y*y'+R;

for i=1:L
    sqrt_w(i)=sqrt(Wc(i));
end
yy = repmat(y, 1, L);
Y1=( Y - yy )*diag(sqrt_w);
[~, R] = qr([ ( Y - yy )*diag(sqrt_w), SR]', 0);

Sp = R;

%P=Y1*diag(Wc)*Y1'+R;


function X=sigmas(x,sr_P,kesi)

%Sigma points around reference point
%Inputs:
%       x: reference point
%       P: covariance
%       c: coefficient
%Output:
%       X: Sigma points
% A = c*chol(P)';
% A=c*P^0.5;
% Y = x(:,ones(1,numel(x)));
% X = [x Y+A Y-A]; 
%P=topdm(P);
%sr_P = sqrtm(P);
%sr_P = chol(P)';

X(1:length(x), 1:length(kesi(1,:)))=0;

for i = 1:length(x)
    for j = 1:length(kesi(1, :))
        X(i, j)=kesi(j, j)*sr_P(i, i)+x(i);            
    end
end

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