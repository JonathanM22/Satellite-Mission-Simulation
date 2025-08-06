clc
clear
% Vraj Patel

% Case 1 Hohmann Transfer to Mars

% constats
au = 149597870.691; 
mu = 1.327e11;

% Dept Orbit (to be changed)
a1 = 1*au; 
e1 = 0; 
f1 = 0; % arbritary

% Arrival Orbit (to be changed)
a2= 1.524*au;
e2 = 0; 
f2 = f1 + pi; 

% transfer orbit
a_transfer = .5*(a1+a2); 
% Time of flight
TOF = pi*sqrt(a_transfer^3/mu); 
fprintf('Hohmann TOF = %.4f days\n', TOF/(3600*24))

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

function [r_vec, v_vec] = orbparm_to_perifocal(a,e,f,mu)

P = a*(1-e^2); 
r = P/(1+e*cos(f));
r_vec = [r*cos(f); r*sin(f)]; 
v_vec = [-sqrt(mu/P)*sin(f); sqrt(mu/P)*(e+cos(f))]; 

end


function Xdot = statespace(mu,X)

x = X(1); 
y = X(2); 
vx = X(3); 
vy = X(4); 

r = sqrt(x^2 + y^2); 
ax = -mu/r^3 * x; 
ay = -mu/r^3 * y; 

Xdot = [vx;vy;ax;ay];

end





