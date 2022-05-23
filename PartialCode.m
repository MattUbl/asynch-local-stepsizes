clear all
close all 
rng(1); % Results in paper were done using rng(3)

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


iter = 1000; %Simulation runtime
r = Q*100*ones(n,1); %Generate r such that the stationary point of the quadratic function is not x = 0
b = 10; %Maximum communication delay between agents
Bc = randi([0 b],n,n); %Communication delays between agents (infrequent communication)
Bc = Bc - diag(diag(Bc)); %Communication delay to self is 0
Bd = randi([0 b],n,n); %Communication delays between agents (infrequent communication)
Bd = Bd - diag(diag(Bd)); %Communication delay to self is 0
L = abs(Q).*(ones(20)+Bc+Bd'); %each entry is L^i_j(1 + D^i_j + D^j_i)
T = ones(20); %Will track \tau^j_i(t)
D = randi([0 4],n);
Dt = ones(20,1);

Xs = zeros(n); %State of global stepsize case
Ms = zeros(n,n,b+1);
Xp = zeros(n); %State of locally chosen stepsize case
Mp = zeros(n,n,b+1);

Gp = (1.9)./sum(L,2); %locally chosen stepsizes
Gs = (1.9)/(norm(Q)*(1+sqrt(n)*20)); % global stepsize

for k = 1:iter
    Up = k*ones(20,1)-D-Dt;
    for i = 1:n
        if Up(i) == 0
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
            Dt(i) = k+1+randi([0 D(i)]);
        end
    end
    Com = k*ones(20)-Bc-T; %A counter for each pair of agents. When Com(i,j) = 0, agent j communicates with agent i. Then T(i,j) is updates such that 0 >= Com(i,j) >= -B(i,j)
    for i = 1:n
        for j = 1:n
            if Com(i,j) == 0 %Probability agent j sends its state to agent i
                g = randi([0 Bd(i,j)]);
                Mp(j,i,g+1) = Xp(j,j); %Agent j shares with agent i
                Ms(j,i,g+1) = Xs(j,j);
                T(i,j) = k+1+randi([0 Bc(i,j)]);
            end
        end
    end
        Rp = diag(Xp); %The actual state of the system, i.e. each agent's copy of its own state
        Rs = diag(Xs);
    ep(k) = 0.5*Rp'*Q*Rp+r'*Rp; %Normalized system error, distance to Sol divided by norm of Sol
    es(k) = 0.5*Rs'*Q*Rs+r'*Rs;
    for l = 1:b
        Mp(:,:,b) = Mp(:,:,b+1);
        Mp(:,:,b+1) = zeros(n);
        Ms(:,:,b) = Ms(:,:,b+1);
        Ms(:,:,b+1) = zeros(n);
        for i = 1:n
            for j = 1:b
                if Ms(i,j,1) ~= 0
                    Xs(i,j) = Ms(i,j,1);
                end
                if Mp(i,j,1) ~= 0
                    Xp(i,j) = Mp(i,j,1);
                end
            end
        end
    end
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
