clc
clear
close all

% --- Mode: choose one of 'inclination' | 'RAAN' | 'AOP'
mode = 'RAAN';  % <<< change here

% --- Constants ---
au = 149597870.691; 
mu = 1.327e11;   % Sun GM (km^3/s^2)

% --- Departure Orbit (Earth-like) ---
a1  = au;
e1  = 0.2;                       % can be non-circular
f1  = deg2rad(0);                % true anomaly at departure
T1  = 2*pi*sqrt(a1^3/mu); 
I1  = deg2rad(60); 
AOP1  = deg2rad(0);             % argument of periapsis (rad)
RAAN1 = deg2rad(30);              % RAAN (rad)

% --- Arrival Orbit (Mars-like) ---
a2  = 1.524*au;
e2  = 0.6;
T2  = 2*pi*sqrt(a2^3/mu);
I2  = deg2rad(60);
AOP2  = deg2rad(0);
RAAN2 = deg2rad(70);

% We will arrive 180 deg after departure (periapsis->apoapsis)
f2_arrival = f1 + pi;

% --- Hohmann-like geometry (periapsis at r1, apogee at r2) ---
r1 = (a1*(1 - e1^2)) / (1 + e1*cos(f1));           % departure radius
r2 = (a2*(1 - e2^2)) / (1 + e2*cos(f2_arrival));   % arrival radius

a_transfer = 0.5*(r1 + r2);
e_transfer = (r2 - r1) / (r2 + r1);
TOF = pi*sqrt(a_transfer^3/mu); 
fprintf('Hohmann-like TOF = %.4f days\n', TOF/(3600*24));

% --- Phase angle / timing (for plotting planets during TOF) ---
E2_arrival   = 2*atan2(tan(f2_arrival/2), sqrt((1+e2)/(1-e2)));
M2_arrival   = E2_arrival - e2*sin(E2_arrival);
M2_departure = M2_arrival - sqrt(mu/a2^3)*TOF; 
E2_departure = Newtonraphson(M2_departure, e2);
f2_start     = 2*atan2( sqrt((1+e2)/(1-e2)) * tan(E2_departure/2), 1);

fprintf('Arrival-planet true anomaly at departure: %.4f deg\n', rad2deg(f2_start));
fprintf('Arrival-planet true anomaly at arrival:   %.4f deg\n', rad2deg(f2_arrival));

phase_angle = f2_start - f1;
if (0 <= rad2deg(phase_angle)) && (rad2deg(phase_angle) <= 180)
    fprintf('Arrival planet leads departure by %.4f deg\n', rad2deg(phase_angle));
else
    fprintf('Arrival planet lags departure by %.4f deg\n', rad2deg(phase_angle));
end

% --- Set transfer-orbit orientation based on selected "pure" change at PERIAPSIS ---
% We apply the element change at the first impulse, so the transfer orbit
% already has the target element value. The other two angular elements stay
% equal to the departure-orbit values.
IT = I1;
RAANt = RAAN1;
AOPt = AOP1;
% defaults

switch lower(mode)
    case 'inclination'   % pure inclination change at periapsis
        IT = I2;         RAANt = RAAN1;       AOPt = AOP1;
    case 'raan'          % pure RAAN change at periapsis
        IT = I1;         RAANt = RAAN2;       AOPt = AOP1;
    case 'aop'           % pure AOP change at periapsis
        IT = I1;         RAANt = RAAN1;       AOPt = AOP2;
    otherwise
        error('Unknown mode: choose ''inclination'', ''RAAN'', or ''AOP''.');
end

% --- Initial/Final state vectors (for propagation) ---
[r1_vec, v1_vec] = orbparm_to_perifocal(a1, e1, f1,mu, I1, RAAN1, AOP1);
state1_vec = [r1_vec; v1_vec];

[r2_vec, v2_vec] = orbparm_to_perifocal(a2, e2, f1+phase_angle, mu, I2, RAAN2, AOP2);
state2_vec = [r2_vec; v2_vec];

% --- Transfer initial state at PERIAPSIS (true anomaly = 0 in its own frame) ---
f_transfer_start = 0;   % periapsis of the transfer ellipse
[rt_vec, vt_vec] = orbparm_to_perifocal(a_transfer, e_transfer, f_transfer_start, mu, IT, RAANt, AOPt);
statet_vec = [rt_vec; vt_vec];

% --- Propagation ---
options = odeset('RelTol',1e-12,'AbsTol',1e-14);

tspan_dep = [0, T1];
[t_dep, X1] = ode45(@(t,X) statespace(t,mu,X), tspan_dep, state1_vec, options);

tspan_arr = [0, T2];
[t_arr, X2] = ode45(@(t,X) statespace(t,mu,X), tspan_arr, state2_vec, options);

tspan_transfer = [0, TOF];
[t_transfer, Xt] = ode45(@(t,X) statespace(t,mu,X), tspan_transfer, statet_vec, options);

% Data up to TOF for the planets
X1_TOF = X1(t_dep <= TOF, :);
X2_TOF = X2(t_arr <= TOF, :);

% --- Plotting ---
figure
plot3(X1(:,1), X1(:,2), X1(:,3), 'r', 'DisplayName','Earth Full Orbit'); hold on
plot3(X2(:,1), X2(:,2), X2(:,3), 'g', 'DisplayName','Mars Full Orbit');
plot3(Xt(:,1), Xt(:,2), Xt(:,3), 'b', 'LineWidth',2, 'DisplayName','Transfer Orbit');

% Planet tracks during TOF
plot3(X1_TOF(:,1), X1_TOF(:,2), X1_TOF(:,3), 'r--', 'LineWidth',2, 'DisplayName','Earth during Transfer');
plot3(X2_TOF(:,1), X2_TOF(:,2), X2_TOF(:,3), 'g--', 'LineWidth',2, 'DisplayName','Mars during Transfer');

% Departure markers
plot3(r1_vec(1), r1_vec(2), r1_vec(3), 'ro', 'MarkerSize',8, 'MarkerFaceColor','r', 'DisplayName','Earth at Departure');
plot3(r2_vec(1), r2_vec(2), r2_vec(3), 'go', 'MarkerSize',8, 'MarkerFaceColor','g', 'DisplayName','Mars at Departure');

% Arrival markers
plot3(X1_TOF(end,1), X1_TOF(end,2), X1_TOF(end,3), 'rs', 'MarkerSize',8, 'MarkerFaceColor','r', 'DisplayName','Earth at Arrival');
plot3(X2_TOF(end,1), X2_TOF(end,2), X2_TOF(end,3), 'gs', 'MarkerSize',8, 'MarkerFaceColor','g', 'DisplayName','Mars at Arrival');

% Transfer endpoints
plot3(Xt(1,1),  Xt(1,2),  Xt(1,3),  'bo', 'MarkerSize',8, 'MarkerFaceColor','b', 'DisplayName','Transfer Start (Periapsis)');
plot3(Xt(end,1), Xt(end,2), Xt(end,3), 'bs', 'MarkerSize',8, 'MarkerFaceColor','b', 'DisplayName','Transfer End (Apogee)');

% Sun
plot3(0,0,0,'o','MarkerSize',12,'MarkerFaceColor','#FFA500','DisplayName','Sun');

title(sprintf('Hohmann-like Transfer (%s change @ periapsis)', upper(mode)))
xlabel('x (km)'); ylabel('y (km)'); zlabel('z (km)');
axis equal; grid on; view(3);
legend('show','Location','best');

% --- Functions -----------------------------------------------------------
function [r_vec, v_vec] = orbparm_to_perifocal(a,e,f,mu,inc,raan,aop)
    P = a*(1-e^2); 
    r = P/(1+e*cos(f));
    % Perifocal (PQW)
    r_pqw = [r*cos(f); r*sin(f); 0];
    v_pqw = [-sqrt(mu/P)*sin(f); sqrt(mu/P)*(e+cos(f)); 0];

    % Rotation matrices: RAAN (about Z), INC (about X'), AOP (about Z'')
    R3_W = [cos(raan), -sin(raan), 0; sin(raan), cos(raan), 0; 0, 0, 1];
    R1_i = [1,0,0; 0, cos(inc), -sin(inc); 0, sin(inc), cos(inc)];
    R3_w = [cos(aop), -sin(aop), 0; sin(aop), cos(aop), 0; 0, 0, 1];

    DCM = R3_W * R1_i * R3_w;   % PQW -> ECI
    r_vec = DCM * r_pqw;
    v_vec = DCM * v_pqw;
end

function Xdot = statespace(~,mu,X)
    r = X(1:3); v = X(4:6);
    a = -mu * r / norm(r)^3;
    Xdot = [v; a];
end

function E = Newtonraphson(M, e)
    E = M;
    for k = 1:20
        f  = E - e*sin(E) - M;
        fp = 1 - e*cos(E);
        dE = -f/fp;
        E  = E + dE;
        if abs(dE) < 1e-12, break; end
    end
end
