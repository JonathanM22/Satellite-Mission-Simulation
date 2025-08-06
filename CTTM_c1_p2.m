clc
clear
% Vraj Patel

% Case 1 part 2 - Inclination change Hohmann transfer

% constats
au = 149597870.691; 
mu = 1.327e11;

% Dept Orbit 
a1 = 1*au; 
e1 = 0; 
f1 = deg2rad(0);
T1 = 2*pi*sqrt(a1^3/mu); 
I1 = deg2rad(50); 

% Arrival Orbit (to be changed)
a2= 1.524*au;
e2 = 0; 
f2 = f1 + pi; 
T2 = 2*pi*sqrt(a2^3/mu); 
I2 = deg2rad(130);

% transfer orbit
a_transfer = .5*(a1+a2); 
e_transfer = (a2-a1)/(a1+a2);  % only for hohmann
alpha = deg2rad(-20); 
beta = I2-I1-alpha;
IT = I1 + alpha; 


% Time of flight
TOF = pi*sqrt(a_transfer^3/mu); 
fprintf('Hohmann TOF = %.4f days\n', TOF/(3600*24))

%% trial min energy eqns derived from Lamberts --> min energy 
% same results --> better for non-hohmann for diff cases --> implement
 
% l = a1+a2; 
% m = a1*a2*(1+cos(df));
% k = a1*a2*(1-cos(df)); 
% a = 1/4*(a1+a2+sqrt(a1^2+a2^2-2*a1*a2*cos(df))); 
% p = (2*a*k*l - k*m)/(2*a*l^2 -4*a*m); 
% e = sqrt(1-p/a);
% disp(e)

%% 

% True anamoly(func of time) = initial t.a + mean motion * (TOF) 
% Hohmann --> Mars final t.a  - Earth initial t.a = pi

phase_angle = pi - sqrt(mu/a2^3)*TOF;
if (0 <= rad2deg(phase_angle)) && (rad2deg(phase_angle) <=180)
    fprintf('Mars leads Earth at Deptarture by %.4f degrees', rad2deg(phase_angle))
else
    fprintf('Mars lags behind Earth at Deptarture by %.4f degrees', rad2deg(phase_angle))
end

% when dealing with non-circular orbits (later) --> EPH Cords
% must do transforms and have function for orb parameters -> pos and vel
% vectors

% initial state vectors
[r1_vec, v1_vec] = orbparm_to_perifocal(a1,e1,f1,mu,I1,0,0);
state1_vec = [r1_vec; v1_vec];
[r2_vec, v2_vec] = orbparm_to_perifocal(a2,e2,f1+phase_angle,mu,I2,0,0);
state2_vec = [r2_vec; v2_vec];
vt = sqrt(2*mu/norm(r1_vec) - mu/a_transfer); 
vt_vec = [-vt*sin(f1);vt*cos(f1);0]; % perifocal
DCM_inc = [ 1 0 0 ; 0 cos(IT) -sin(IT) ; 0 sin(IT) cos(IT)]; %peri to eci
vt_vec = DCM_inc * vt_vec;
fprintf('\n vt_vec = [ %.4f %.4f %.4f ]',vt_vec(1), vt_vec(2), vt_vec(3))
statet_vec = [r1_vec; vt_vec];

% rotate sat velocity vector same amount as Earth true anomaly: cord transfrom
% earth and mars have same perifocal frame

options = odeset('RelTol',1e-12,'AbsTol',1e-14);
tspan_dep = [0,T1];
[t_dep,X1] = ode45(@(t,X) statespace(t,mu,X), tspan_dep, state1_vec,options);
tspan_arr = [0,T2]; 
[t_arr,X2] = ode45(@(t,X) statespace(t,mu,X), tspan_arr, state2_vec,options);

% if it weren't hohmann, would have to find the TSPAN using E, f and M 
% not newton raphson

tspan_transfer = [0,TOF];
[t_transfer,Xt] = ode45(@(t,X) statespace(t, mu,X), tspan_transfer, statet_vec ,options);

% Use find to extract Earth and Mars data up to TOF
X1_TOF = X1(t_dep <= TOF, :);
X2_TOF = X2(t_arr <= TOF, :);

figure 
plot3(X1(:,1),X1(:,2),X1(:,3),'r')
hold on 
plot3(X2(:,1),X2(:,2),X2(:,3),'g')
plot3(Xt(:,1),Xt(:,2),Xt(:,3),'b')

% Plot planet during TOF 
plot3(X1_TOF(:,1),X1_TOF(:,2),X1_TOF(:,3),'r--','LineWidth',2)
plot3(X2_TOF(:,1),X2_TOF(:,2),X2_TOF(:,3),'g--','LineWidth',2)
%  initial positions
plot3(r1_vec(1),r1_vec(2),r1_vec(3),'ro','MarkerSize',8,'MarkerFaceColor','r')
plot3(r2_vec(1),r2_vec(2),r2_vec(3),'go','MarkerSize',8,'MarkerFaceColor','g')
% Mark final positions at end of TOF
plot3(X1_TOF(end,1),X1_TOF(end,2),X1_TOF(end,3),'rs','MarkerSize',8,'MarkerFaceColor','r')
plot3(X2_TOF(end,1),X2_TOF(end,2),X2_TOF(end,3),'gs','MarkerSize',8,'MarkerFaceColor','g')
%Add Sun at origin
plot3(0, 0, 0,'o', 'MarkerSize', 12, 'MarkerFaceColor', '#FFA500');
title('Hohmann Transfer to Mars')
xlabel('X Km')
ylabel('Y Km')
zlabel('Z Km')
axis("equal")
%% 

function [r_vec, v_vec] = orbparm_to_perifocal(a,e,f,mu,in,raan,aop)
P = a*(1-e^2); 
r = P/(1+e*cos(f));
% perifocal cords (EPH)
r_vec = [r*cos(f); r*sin(f);0]; 
v_vec = [-sqrt(mu/P)*sin(f); sqrt(mu/P)*(e+cos(f));0]; 
% for plane change --> Need DCM's to transform from Perifocal to ECI 

DCM_inc = [ 1 0 0 ; 0 cos(in) -sin(in)  ; 0 sin(in) cos(in) ];
% DCM_RAAN = [cos(raan) -sin(raan) 0 ; sin(raan) cos(raan) 0 ; 0 0 1]; 
% DCM_AOP = [cos(aop) -sin(aop) 0 ; sin(aop) cos(aop) 0 ; 0 0 1]; 

% DCM = DCM_AOP * DCM_inc * DCM_RAAN; 

r_vec = DCM_inc * r_vec;
v_vec = DCM_inc * v_vec;

end

function Xdot = statespace(~,mu,X)
x = X(1); 
y = X(2); 
z = X(3);
vx = X(4); 
vy = X(5); 
vz = X(6);
r = sqrt(x^2 + y^2 +z^2); 
ax = -mu/r^3 * x; 
ay = -mu/r^3 * y; 
az = -mu/r^3 * z;
Xdot = [vx;vy;vz;ax;ay;az];
end