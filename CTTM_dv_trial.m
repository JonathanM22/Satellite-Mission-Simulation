clc
clear
% Vraj Patel

% Case 1 part 3 Shape Change (Eccentricity): Earth --> Mars
% Future Work --> DElta V

% % constats
au = 149597870.691; 
mu = 1.327e11;

% Dept Orbit (to be changed)
a1 = au;
e1 = 0;
f1 = deg2rad(30); % arbritary
T1 = 2*pi*sqrt(a1^3/mu); 

% Arrival Orbit (to be changed)
a2= 1.524*au;
e2 = 0; 
f2_arrival = f1 + pi; % Where Mars should be when satellite arrives
T2 = 2*pi*sqrt(a2^3/mu); 


% Fun trial for validation
% -------------------------------------------------------------------------
% single impulse shape change Hohmann --> TLI works and outputs right DV

% mu =3.986e5; 
% a1 = 6378+185; 
% e1 = 0; 
% f1 = pi; 
% T1 = 2*pi*sqrt(a1^3/mu); 
% r_p1 = a1*(1-e1);
% 
% ar2 = 384400; 
% a2 = .5*(a1+ar2); 
% e2 = 1-ar2/a2;
% f2_arrival = f1 + pi;
% T2 = 2*pi*sqrt(a2^3/mu); 
% -------------------------------------------------------------------------


% transfer orbit
% -------------------------------------------------------------------------
% trial min energy eqns derived from Lamberts --> min energy 
% same results --> better for non-hohmann for diff cases --> implement

df = pi; % 180 degree transfer
p1 = a1*(1-e1^2); 
p2 = a2*(1-e2^2); 
r1 = p1/(1+e1*cos(f1));
r2 = p2/(1+e2*cos(f1 + pi)); % Mars position when satellite arrives
l = r1+r2; 
m = r1*r2*(1+cos(df));
k = r1*r2*(1-cos(df)); 
a_transfer = 1/4*(r1+r2+sqrt(r1^2+r2^2-2*r1*r2*cos(df))); 
p = ( 2*a_transfer*k*l -k*m )/( 2*a_transfer*l^2 - 4*a_transfer*m); 
e_transfer = sqrt(1-p/a_transfer);
%--------------------------------------------------------------------------

% Time of flight
TOF = pi*sqrt(a_transfer^3/mu); 
fprintf('Hohmann TOF = %.4f days\n', TOF/(3600*24))

% when dealing with non-circular orbits (later) --> ECI Cords
% must do transforms and have function for orb parameters -> pos and vel
% vectors

% Calculate Mars true anomaly at departure (f2_start)
E2_arrival = 2*atan2(tan(f2_arrival/2),sqrt((1+e2)/(1-e2)));
M2_arrival = E2_arrival-e2*sin(E2_arrival); 
M2_departure = M2_arrival-sqrt(mu/a2^3)*TOF; 
E2_departure = Newtonraphson(M2_departure, e2);
f2_start = 2*atan2(sqrt((1+e2)/(1-e2)) *tan(E2_departure/2),1);

fprintf('Mars true anomaly at departure: %.4f degrees\n', rad2deg(f2_start))
fprintf('Mars true anomaly at arrival: %.4f degrees\n', rad2deg(f2_arrival))

phase_angle = f2_start - f1;
if (0 <= rad2deg(phase_angle)) && (rad2deg(phase_angle) <=180)
    fprintf('Mars leads Earth at Deptarture by %.4f degrees', rad2deg(phase_angle))
else
    fprintf('Mars lags behind Earth at Deptarture by %.4f degrees', rad2deg(phase_angle))
end

% initial state vectors

% Departure Orbit
[r1_vec, v1_vec] = orbparm_to_perifocal(a1,e1,f1,mu);
state1_vec = [r1_vec; v1_vec];

% Arrival Orbit
[r2_vec, v2_vec] = orbparm_to_perifocal(a2,e2,f2_start,mu);
state2_vec = [r2_vec; v2_vec];

% rotate sat velocity vector same amount as Earth true anomaly: cord transfrom
% earth and mars have same perifocal frame

% Transfer Orbit
vt = sqrt(2*mu/norm(r1_vec) - mu/a_transfer); 
vt_vec = [-vt*sin(f1);vt*cos(f1);0];
statet_vec = [r1_vec; vt_vec];

% Numerical Integration
options = odeset('RelTol',1e-12,'AbsTol',1e-14);
tspan_dep = [0,T1];
[t_dep,X1] = ode45(@(t,X) statespace(t,mu,X), tspan_dep, state1_vec,options);
tspan_arr = [0,T2]; 
[t_arr,X2] = ode45(@(t,X) statespace(t,mu,X), tspan_arr, state2_vec,options);
tspan_transfer = [0,TOF];
[t_transfer,Xt] = ode45(@(t,X) statespace(t, mu,X), tspan_transfer, statet_vec ,options);

% Use find to extract Earth and Mars data up to TOF
X1_TOF = X1(t_dep <= TOF, :);
X2_TOF = X2(t_arr <= TOF, :);

% Plotting
figure 
hold on 
plot(X1(:,1),X1(:,2),'r','DisplayName','Earth Full Orbit')
plot(X2(:,1),X2(:,2),'g','DisplayName','Mars Full Orbit')
plot(Xt(:,1),Xt(:,2),'b','DisplayName','Transfer Orbit')

% Plot planet positions during TOF 
plot(X1_TOF(:,1),X1_TOF(:,2),'r--','LineWidth',2,'DisplayName','Earth during Transfer')
plot(X2_TOF(:,1),X2_TOF(:,2),'g--','LineWidth',2,'DisplayName','Mars during Transfer')

% Mark initial positions (at departure)
plot(r1_vec(1),r1_vec(2),'ro','MarkerSize',8,'MarkerFaceColor','r','DisplayName','Earth at Departure')
plot(r2_vec(1),r2_vec(2),'go','MarkerSize',8,'MarkerFaceColor','g','DisplayName','Mars at Departure')

% Mark final positions at end of TOF
plot(X1_TOF(end,1),X1_TOF(end,2),'rs','MarkerSize',8,'MarkerFaceColor','r','DisplayName','Earth at Arrival')
plot(X2_TOF(end,1),X2_TOF(end,2),'gs','MarkerSize',8,'MarkerFaceColor','g','DisplayName','Mars at Arrival')

% % Add Sun at origin
% plot(0, 0, 'o', 'MarkerSize', 12, 'MarkerFaceColor', '#FFA500', 'DisplayName', 'Sun');

title('Hohmann Transfer to Mars')
xlabel('Distance (km)')
ylabel('Distance (km)')
legend('show')
axis equal
grid on

deltaV = deltaV_calc(r1,r2,a1,a2,a_transfer,e1,e2,e_transfer,f1,f2_arrival,mu);
fprintf('\nTotal Delta V required for Transfer = %.4f km/s',deltaV)
% Functions ---------------------------------------------------------------

function deltaV = deltaV_calc(r1,r2,a1,a2,at,e1,e2,et,f1,f2,mu)  % works only for hohmann
v1 = sqrt(2*mu/r1 - mu/a1);
vt1 = sqrt(2*mu/r1 -mu/at);
g1 = atan2(e1*sin(f1),(1+e1*cos(f1))); % flight path angle at deptarture orbit 1
Pt = at*(1-et^2); % semi latus rectum of transfer ellipse
ft1 = acos(((Pt/r1)-1)/et); % true anomaly of transfer ellipse at departure
if real(ft1) == 0 || real(ft1) == pi
    gt1 = 0; 
else 
gt1 = atan2(et*sin(ft1),(1+et*cos(ft1))); % flight path angle of T.E at dept
end
dg1 = g1-gt1;
dv1 = sqrt(v1^2 + vt1^2 - 2*v1*vt1*cos(dg1)); 
vt2 = sqrt(2*mu/r2 -mu/at);
v2 = sqrt(2*mu/r2 -mu/a2);
g2 = atan2(e2*sin(f2),(1+e2*cos(f2))); % flight path angle at arrival orbit 2
ft2 = acos(((Pt/r2)-1)/et); % true anomaly of transfer ellipse at arrival
if real(ft2) == 0 || real(ft2) == pi
    gt2 = 0;
else
gt2 = atan2(et*sin(ft2),(1+et*cos(ft2))); % flight path angle of T.E at arrival
end
dg2 = gt2-g2;
dv2 = sqrt(v2^2 + vt2^2 - 2*v2*vt2*cos(dg2)); 
fprintf('\nDelta V from Orbit 1 to Transfer orbit is %.4f Km/s', dv1); 
fprintf('\nDelta V from Transfer Orbit to Orbit 2 is %.4f Km/s', dv2); 
deltaV = dv1 + dv2; 
end


function [r_vec, v_vec] = orbparm_to_perifocal(a,e,f,mu)
    P = a*(1-e^2); 
    r = P/(1+e*cos(f));
    r_vec = [r*cos(f); r*sin(f); 0]; 
    v_vec = [-sqrt(mu/P)*sin(f); sqrt(mu/P)*(e+cos(f)); 0]; 
end

function Xdot = statespace(~,mu,X)
    r_vec = X(1:3);
    v_vec = X(4:6);
    r = norm(r_vec);
    a_vec = -mu / r^3 * r_vec;
    Xdot = [v_vec; a_vec];
end

function E = Newtonraphson(M, e)
    E = M;  % Initial guess
    for i = 1:10 
        E_new = E - (E - e*sin(E) - M) / (1 - e*cos(E));
        if abs(E_new - E) < 1e-12
            break;
        end
        E = E_new;
    end
end