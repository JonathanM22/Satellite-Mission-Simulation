clc
clear
% Vraj Patel

% Case 1 Hohmann Transfer to Mars

% Constants
au = 149597870.691; 
mu = 1.327e11;   

% Dept Orbit (to be changed)
a1 = 1*au; 
e1 = 0; 
f1 = deg2rad(40); % arbritary

% Arrival Orbit (to be changed)
a2= 1.524*au;
e2 = 0; 
f2 = f1 + pi; 

hohmann_transfer(a1, e1, f1,f2, a2, e2, mu)
%% 

function [r_vec, v_vec] = orbparm_to_perifocal(a,e,f,mu)
P = a*(1-e^2); 
r = P/(1+e*cos(f));
r_vec = [r*cos(f); r*sin(f)]; 
v_vec = [-sqrt(mu/P)*sin(f); sqrt(mu/P)*(e+cos(f))]; 
end

function Xdot = statespace(~,mu,X)
x = X(1); 
y = X(2); 
vx = X(3); 
vy = X(4); 
r = sqrt(x^2 + y^2); 
ax = -mu/r^3 * x; 
ay = -mu/r^3 * y; 
Xdot = [vx;vy;ax;ay];
end

function hohmann_transfer(a1, e1, f1,f2, a2, e2, mu)
 
    % Orbital periods
    T1 = 2*pi*sqrt(a1^3/mu); 
    T2 = 2*pi*sqrt(a2^3/mu); 

    % Transfer orbit parameters
    a_transfer = 0.5 * (a1 + a2); 

    %trial min energy eqns derived from Lamberts --> min energy 
    %same results --> better for non-hohmann for diff cases --> implement
     
    df = f2 - f1;
    l = a1+a2; 
    m = a1*a2*(1+cos(df));
    k = a1*a2*(1-cos(df)); 
    % a = (m*l*p)/((2*m-l^2)*p^2+2*k*l*p-k^2);
    a = 1/4*(a1+a2+sqrt(a1^2+a2^2-2*a1*a2*cos(df))); 
    p = (2*a*k*l - k*m)/(2*a*l^2 -4*a*m); 
    e = sqrt(1-p/a); 
    disp(e)

    e_transfer = (a2 - a1) / (a1 + a2); 
    disp(e_transfer)
    TOF = pi * sqrt(a_transfer^3 / mu); 

    fprintf('Hohmann TOF = %.4f days\n', TOF / (3600*24))

    % Phase angle calculation
    phase_angle = pi - sqrt(mu/a2^3)*TOF;
    if (0 <= rad2deg(phase_angle)) && (rad2deg(phase_angle) <= 180)
        fprintf('Mars leads Earth at Departure by %.4f degrees\n', rad2deg(phase_angle))
    else
        fprintf('Mars lags behind Earth at Departure by %.4f degrees\n', rad2deg(phase_angle))
    end

    % Initial/final state vectors
    [r1_vec, v1_vec] = orbparm_to_perifocal(a1, e1, f1, mu);
    state1_vec = [r1_vec; v1_vec];
    [r2_vec, v2_vec] = orbparm_to_perifocal(a2, e2, f1 + phase_angle, mu);
    state2_vec = [r2_vec; v2_vec];

    % Transfer orbit velocity vector
    vt = sqrt(2*mu/norm(r1_vec) - mu/a_transfer); 
    vt_vec = [-vt*sin(f1); vt*cos(f1)];
    statet_vec = [r1_vec; vt_vec];

    % Propagate using ODE
    options = odeset('RelTol',1e-12,'AbsTol',1e-14);
    [t_dep, X1] = ode45(@(t,X) statespace(t,mu,X), [0, T1], state1_vec, options);
    [t_arr, X2] = ode45(@(t,X) statespace(t,mu,X), [0, T2], state2_vec, options);
    [t_transfer, Xt] = ode45(@(t,X) statespace(t, mu,X), [0, TOF], statet_vec, options);

    % Trim for TOF
    X1_TOF = X1(t_dep <= TOF, :);
    X2_TOF = X2(t_arr <= TOF, :);
    figure 
    hold on 
    plot(X1(:,1),X1(:,2),'r')
    plot(X2(:,1),X2(:,2),'g')
    plot(Xt(:,1),Xt(:,2),'b')
    
    % Plot planet during TOF 
    plot(X1_TOF(:,1),X1_TOF(:,2),'r--','LineWidth',2)
    plot(X2_TOF(:,1),X2_TOF(:,2),'g--','LineWidth',2)
    %  initial positions
    plot(r1_vec(1),r1_vec(2),'ro','MarkerSize',8,'MarkerFaceColor','r')
    plot(r2_vec(1),r2_vec(2),'go','MarkerSize',8,'MarkerFaceColor','g')
    % Mark final positions at end of TOF
    plot(X1_TOF(end,1),X1_TOF(end,2),'rs','MarkerSize',8,'MarkerFaceColor','r')
    plot(X2_TOF(end,1),X2_TOF(end,2),'gs','MarkerSize',8,'MarkerFaceColor','g')
    % Add Sun at origin
    plot(0, 0, 'o', 'MarkerSize', 12, 'MarkerFaceColor', '#FFA500');
    title('Hohmann Transfer to Mars')
    xlabel('Km')
    ylabel('Km')
    axis("equal")
end


