%
% Batch and recursive linear regression. Uses
% cell evaluation.
%

%%
% Simulate data
%

  randn('state',12);
  dt = 0.01;
  sd = 0.1;
  t = (0:dt:1);
  x = 1 + 0.5*t;
  y = x + sd*randn(size(x));

  clf;
  h = plot(t,y,'.',t,x);
  
  axis([0 1 0.5 2]);
  
  set(h,'Markersize',7);
  set(h,'LineWidth',4);
  set(h(1),'Color',[0.0 0.0 0.0]);
  set(h(2),'Color',[0.7 0.7 0.7]);
  
  
  h = legend('Measurement','True Signal');
  xlabel('{\it t}');
  ylabel('{\it y}');

%%
% Batch linear regression
%
  m0 = [0;0];
  P0 = 1*eye(2);
  n  = length(y);
  % Hk = [1 tk]
  H  = [ones(length(t),1) t'];
  mb = inv(inv(P0) + 1/sd^2*H'*H)*(1/sd^2*H'*y'+inv(P0)*m0)
  Pb = inv(inv(P0) + 1/sd^2*H'*H);
  
  clf;
  h = plot(t,y,'.',t,x,t,mb(1)+mb(2)*t,'-');

  axis([0 1 0.5 2]);
  
  set(h,'Markersize',7);
  set(h(2),'LineWidth',4);
  set(h(3),'LineWidth',1.5);
  set(h(1),'Color',[0.0 0.0 0.0]);
  set(h(2),'Color',[0.7 0.7 0.7]);
  set(h(3),'Color',[0.0 0.0 0.0]);
  
  h = legend('Measurement','True Signal','Estimate');
  xlabel('{\it t}');
  ylabel('{\it y}');
  
%%
% Kalman filter
%
  m = m0;
  P = P0;
  MM = zeros(size(m0,1),length(y));
  PP = zeros(size(P0,1),size(P0,1),length(y));
  count = 0;
  for k=1:length(y)
      H = [1 t(k)];
      S = H*P*H'+sd^2;
      K = P*H'/S;
      m = m + K*(y(k)-H*m);
      P = P - K*S*K';
%      m = inv(inv(P) + 1/sd^2*H'*H)*(1/sd^2*H'*y(k)+inv(P)*m)
%      P = inv(inv(P) + 1/sd^2*H'*H);
      
      MM(:,k) = m;
      PP(:,:,k) = P;

      HH = [ones(length(t),1) t'];
      VV = diag(HH*P*HH');
      q1 = HH*m+1.96*sqrt(VV);
      q2 = HH*m-1.96*sqrt(VV);
      
      clf;
      p=patch([t fliplr(t)],[q1' fliplr(q2')],1);
      set(p,'FaceColor',[0 1 0])
      hold on;
      h = plot(t,y,'.',t,x,t,HH*m,'-',...
               t(1:k),y(1:k),'ko');

      axis([0 1 0.5 2]);
  
      set(h,'linewidth',2);
      set(h,'Markersize',10);
      set(h(2),'LineWidth',4);
      set(h(3),'LineWidth',1.5);
      set(h(4),'Markersize',10);
      grid on;
      
      pause(0.1);
      drawnow;
  end
  m

%%  
% Plot the evolution of estimates
%

  clf;
  h = plot(t,MM(1,:),'b-',[0 1],[mb(1) mb(1)],'b--',...
           t,MM(2,:),'r-',[0 1],[mb(2) mb(2)],'r--');
  
  set(h,'Markersize',10);
  set(h,'LineWidth',2);
  set(h(1),'Color',[0.0 0.0 0.0]);
  set(h(2),'Color',[0.0 0.0 0.0]);
  set(h(3),'Color',[0.5 0.5 0.5]);
  set(h(4),'Color',[0.5 0.5 0.5]);

  
  h = legend('Recursive E[ {\it\theta}_1 ]','Batch E[ {\it\theta}_1 ]',...
         'Recursive E[ {\it\theta}_2 ]','Batch E[ {\it\theta}_2 ]',4);
  
  xlabel('{\it t}');
  ylabel('{\it y}');

%%  
% Plot the evolution of variances
%

  clf;
  h = semilogy(t,squeeze(PP(1,1,:)),'b-',[0 1],[Pb(1,1) Pb(1,1)],'b--',...
               t,squeeze(PP(2,2,:)),'r-',[0 1],[Pb(2,2) Pb(2,2)],'r--');

  set(h,'Markersize',10);
  set(h,'LineWidth',2);
  set(h(1:2),'Color',[0.0 0.0 0.0]);
  set(h(3:4),'Color',[0.5 0.5 0.5]);
  
  h = legend('Recursive Var[ {\it\theta}_1 ]','Batch Var[ {\it\theta}_1 ]',...
         'Recursive Var[ {\it\theta}_2 ]','Batch Var[ {\it\theta}_2 ]');     
  
  xlabel('{\it t}');
  ylabel('{\it y}');
  grid on;
  
