clear all
close all 
rng(2); % Results in paper were done using rng(1)

%%% Generate quadratic function
n = 20; %matrix size
a = randn(n,n);
eig_mean = 1;
a = a'*diag(abs(eig_mean+randn(n,1)))*a;
a = 0.5*a+a';              % symmetric with random entries beween -2 and 2
C = 1000;     % eig(Q)        % desired condition number
[u s v] = svd(a);
s = diag(s);           % s is vector
% ===== linear stretch of existing s
s = s(1)*( 1-((C-1)/C)*(s(1)-s)/(s(1)-s(end))) ;
for i = 1:20 %Makes Q indefinite
    coin = rand();
    if coin > .5
        s(i) = s(i);
    else
        s(i) = -s(i);
    end
end
% =====
s = diag(s);           % back to matrix
Q = u*s*v';
Q = 100*Q/(norm(Q));


iter = 500; %Simulation runtime
r = Q*100*ones(n,1); %Generate r such that the stationary point of the quadratic function is not x = 0
b = 20; %Maximum communication delay between agents
B = randi([0 b],n,n); %Communication delays between agents
B = B - diag(diag(B)); %Communication delay to self is 0
L = abs(Q).*(ones(20)+B+B'); %each entry is L^i_j(1 + D^i_j + D^j_i)
T = ones(20); %Will track \tau^j_i(t)

Xs = zeros(n); %State of global stepsize case
Xp = zeros(n); %State of locally chosen stepsize case

Gp = (1.9)./sum(L,2); %locally chosen stepsizes
Gs = (1.9)/(norm(Q)*(1+sqrt(n)*20)); % global stepsize

for k = 1:iter
    for i = 1:n
            Xp(i,i) = Xp(i,i)-Gp(i)*(Q(i,:)*Xp(:,i)+r(i)); %Locally chosen stepsize update, projected onto X
            if Xp(i,i) > 10000
                Xp(i,i) = 10000;
            elseif Xp(i,i) < -10000
                Xp(i,i) = -10000;
            end
            Xs(i,i) = Xs(i,i)-Gs*(Q(i,:)*Xs(:,i)+r(i)); %Global stepsize update, projected onto X
            if Xs(i,i) > 10000
                Xs(i,i) = 10000;
            elseif Xs(i,i) < -10000
                Xs(i,i) = -10000;
            end
    end
    Com = k*ones(20)-B-T; %A counter for each pair of agents. When Com(i,j) = 0, agent j communicates with agent i. Then T(i,j) is updates such that 0 >= Com(i,j) >= -B(i,j)
    for i = 1:n
        for j = 1:n
            if Com(i,j) == 0 %Probability agent j sends its state to agent i
                Xp(j,i) = Xp(j,j); %Agent j shares with agent i
                Xs(j,i) = Xs(j,j);
                T(i,j) = k+1+randi([0 B(i,j)]);
            end
        end
    end
        Rp = diag(Xp); %The actual state of the system, i.e. each agent's copy of its own state
        Rs = diag(Xs);
    ep(k) = 0.5*Rp'*Q*Rp+r'*Rp; %Normalized system error, distance to Sol divided by norm of Sol
    es(k) = 0.5*Rs'*Q*Rs+r'*Rs;
end

figure(1)
semilogy(ep','LineWidth',2)
hold on
semilogy(es','--','LineWidth',2)
title('Cost Convergence Comparison')
xlabel('Iteration Number','FontWeight','Bold')
ylabel('System Cost','FontWeight','Bold')
legend({'Locally Chosen','Global'})
hold off