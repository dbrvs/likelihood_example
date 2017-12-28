%poisson likelihood
clear; clf

%simulate data with a given distribution parameter

lam=2;
ns=1000;

hist_data = histogram(poissrnd(lam,[1,ns]));

d_x=hist_data.Values;

x=0:length(d_x)-1;

px_data = d_x/sum(d_x);

% then try just maximize likelihood
naive_lam = sum(d_x.*(0:x(end)))/sum(d_x);

in1=1;
nl=1e4;
lam_theory = linspace(0.1,5,nl);
logL=zeros([1,nl]);

for l=lam_theory;
    
    px_theory = poisspdf(x,l);
    %logL(in1)=sum(log(px_theory));
    logL(in1)=sum(d_x.*log(px_theory));
    
    in1=in1+1;
    
end

in_max_lam = find(logL>=max(logL));

%
% or try expect max (EM) algorithm
% 
% tol=1;
% lam_em1=1;
% while abs(tol)>1e-10
%     
%     px_theory = poisspdf(x,lam_em1); %compute the conditional distributino
%     
%     in1=1;
%     for l=lam_theory
%         L_temp = d_x.*log(poisspdf(x,l));
%         
%         Q(in1) = sum(px_theory.*L_temp);
%         in1=in1+1;
%     end
%         
%     in_max_lam = find(Q>=max(Q)); %find max L
%     
%     lam_em2=lam_theory(in_max_lam);
%     
%     tol=lam_em2-lam_em1;
% 
%     disp(tol)
%     
%     lam_em1=lam_em2;
% 
% end
%% PLOT
figure()
   
hold on
plot(lam_theory,logL)
scatter(lam,max(logL),100)
scatter(naive_lam,max(logL),'FillColor','g')
scatter(lam_theory(in_max_lam),logL(in_max_lam),10,'FillColor','k')
%scatter(lam_theory(in_max_lam),logL(in_max_lam),10,'FillColor','r')

legend('log likelihood','true \theta','naive \theta','Bayes \theta')%,'EM')
hold off

title([num2str(ns) ' samples'])

xlabel('poisson parameter')
ylabel('log-likelihood')

print -dpdf poisstest.pdf
    