clc
clear 
close all

% Combined code 

% constants
au = 149597870.691; 
mu = 1.327e11;
% mu =3.986e5; % just to test known examples

% --- Departure Orbit (Earth-like) ---
a1  = au;
e1  = 0.8;                       % can be non-circular
f1  = deg2rad(0);                % true anomaly at departure
T1  = 2*pi*sqrt(a1^3/mu); 
I1  = deg2rad(30);
AOP1  = deg2rad(0);             % argument of periapsis (rad)
RAAN1 = deg2rad(20);              % RAAN (rad)

% --- Arrival Orbit (Mars-like) ---
a2  = 1.524*au;
e2  = 0.6;
f2 = 0; 
T2  = 2*pi*sqrt(a2^3/mu);
I2  = deg2rad(30);
AOP2  = deg2rad(0);
RAAN2 = deg2rad(40);

% Fun trial for validation
% -------------------------------------------------------------------------
% single impulse shape change Hohmann --> TLI works and outputs right DV
% 
% mu =3.986e5; 
% a1 = 6378+185; 
% e1 = 0; 
% f1 = pi; 
% T1 = 2*pi*sqrt(a1^3/mu); 
% r_p1 = a1*(1-e1);
% I1 = deg2rad(0); 
% AOP1 = rad2deg(0); 
% RAAN1 = rad2deg(0); 
% 
% 
% ar2 = 384400; 
% a2 = .5*(a1+ar2); 
% e2 = 1-ar2/a2;
% f2_arrival = f1 + pi;
% T2 = 2*pi*sqrt(a2^3/mu); 
% I2 = deg2rad(0);
% AOP2 = rad2deg(0); 
% RAAN2 = rad2deg(0);
% -------------------------------------------------------------------------

Hohmann_like_transfer(a1,a2,e1,e2,f1,I1,I2,RAAN1, RAAN2, AOP1, AOP2,mu)

% Functions ---------------------------------------------------------------
function Hohmann_like_transfer(a1,a2,e1,e2,f1,I1,I2,RAAN1, RAAN2, AOP1, AOP2,mu)

aop_d = (AOP2+RAAN2) - (AOP1+RAAN1)

aop_dcm = [  cos(aop_d), sin(aop_d),  0;
            -sin(aop_d), cos(aop_d),  0;
                  0,          0,      1];

T1 = 2*pi*sqrt(a1^3/mu); 
T2 = 2*pi*sqrt(a2^3/mu); 


f2_arrival = f1 + pi - aop_d; % Where Mars should be when satellite arrives

% transfer orbit
% -------------------------------------------------------------------------
% trial min energy eqns derived from Lamberts --> min energy 
% same results --> better for non-hohmann for diff cases --> implement

df = pi; % 180 degree transfer
p1 = a1*(1-e1^2); 
p2 = a2*(1-e2^2); 

r1 = p1/(1+e1*cos(f1))
r2_1 = p2/(1+e2*cos(f1 + pi - aop_d)); % Mars position when satellite arrives

r1_eph = [r1*cos(f1); r1*sin(f1); 0];
r2_eph = [r2_1*cos(f1 + pi - aop_d); r2_1*sin(f1 + pi - aop_d); 0]; 
df_test = rad2deg(acos(dot(r1_eph, r2_eph) / (r1*r2_1)))

r2 = r2_1;
l = r1+r2; 
m = r1*r2*(1+cos(df))py;
k = r1*r2*(1-cos(df)); 
a_transfer = 1/4*(r1+r2+sqrt(r1^2+r2^2-2*r1*r2*cos(df))); 
p = ( 2*a_transfer*k*l -k*m )/( 2*a_transfer*l^2 - 4*a_transfer*m); 
e_transfer = sqrt(1-p/a_transfer);

RAANt = RAAN1;
AOPt = AOP1;

if I1 ~= I2

    a = -pi/2:pi/180:pi/2;
    n = length(a); 

    % preallocate for speed    
    deltaV_array = zeros(1,n); 
    alpha_array = zeros(1,n); 
    IT_array = zeros(1,n);  

    for i = 1:n
        alpha_array(i) = a(i);
        beta = I2 - I1 - alpha_array(i);
        IT_array(i) = I1 + alpha_array(i);
        deltaV_array(i) = deltaV_calc(r1, r2, a1, a2, a_transfer, e1, e2, e_transfer,f1, f2_arrival, I1, I2, alpha_array(i), beta, mu);
    end
    [min_dv, min_idx] = min(deltaV_array);
    alpha = alpha_array(min_idx);
    IT = IT_array(min_idx);
    fprintf('\nMinimum Delta V of %.5f km/s achieved at Alpha = %.5f degrees: Transfer Ellipse Inclination at %.5f degrees', min_dv ,rad2deg(alpha), rad2deg(IT))

    % % if u want to pick ur own alpha: 
    % alpha = deg2rad(-20); %Arbritary
    % beta = I2-I1-alpha;
    % IT = I1 + alpha; 

elseif RAAN1 ~= RAAN2
    IT = I1;         RAANt = RAAN2;       AOPt = AOP1;
elseif AOP1 ~= AOP2
    IT = I1;         RAANt = RAAN1;       AOPt = AOP2;
else 
    alpha = 0;
    beta = I2-I1-alpha;
    IT = I1 + alpha; 
    deltaV = deltaV_calc(r1,r2,a1,a2,a_transfer,e1,e2,e_transfer,f1,f2_arrival,I1,I2,alpha,beta,mu);
    
end

%--------------------------------------------------------------------------

% Time of flight
TOF = pi*sqrt(a_transfer^3/mu); 
fprintf('\nHohmann TOF = %.4f days\n', TOF/(3600*24))

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
[r1_vec, v1_vec] = orbparm_to_ECI(a1,e1,f1,mu,I1,RAAN1,AOP1);
state1_vec = [r1_vec; v1_vec];

% Arrival Orbit
[r2_vec, v2_vec] = orbparm_to_ECI(a2,e2,f2_start,mu,I2,RAAN2,AOP2);
state2_vec = [r2_vec; v2_vec];

% rotate sat velocity vector same amount as Earth true anomaly: cord transfrom
% earth and mars have same perifocal frame

% --- Transfer initial state at PERIAPSIS (true anomaly = 0 in its own frame) ---
f_transfer_start = 0;   % periapsis of the transfer ellipse
[rt_vec, vt_vec] = orbparm_to_ECI(a_transfer, e_transfer, f_transfer_start, mu, IT, RAANt, AOPt);
statet_vec = [rt_vec; vt_vec];

% Numerical Integration
options = odeset('RelTol',1e-12,'AbsTol',1e-14);
tspan_dep = [0,T1];
[t_dep,X1] = ode45(@(t,X) statespace(t,mu,X), tspan_dep, state1_vec,options);
tspan_arr = [0,T2]; 
[t_arr,X2] = ode45(@(t,X) statespace(t,mu,X), tspan_arr, state2_vec,options);
tspan_transfer = [0,TOF];
[~,Xt] = ode45(@(t,X) statespace(t, mu,X), tspan_transfer, statet_vec ,options);

% Use find to extract Earth and Mars data up to TOF
X1_TOF = X1(t_dep <= TOF, :);
X2_TOF = X2(t_arr <= TOF, :);

% Plotting
figure 
plot3(X1(:,1),X1(:,2),X1(:,3),'r','DisplayName','Earth Full Orbit')
hold on 
plot3(X2(:,1),X2(:,2),X2(:,3),'g','DisplayName','Mars Full Orbit')
plot3(Xt(:,1),Xt(:,2),Xt(:,3),'b','DisplayName','Transfer Orbit')

% Plot planet positions during TOF 
plot3(X1_TOF(:,1),X1_TOF(:,2),X1_TOF(:,3),'r--','LineWidth',2,'DisplayName','Earth during Transfer')
plot3(X2_TOF(:,1),X2_TOF(:,2),X2_TOF(:,3),'g--','LineWidth',2,'DisplayName','Mars during Transfer')

% Mark initial positions (at departure)
plot3(r1_vec(1),r1_vec(2),r1_vec(3),'ro','MarkerSize',8,'MarkerFaceColor','r','DisplayName','Earth at Departure')
plot3(r2_vec(1),r2_vec(2),r2_vec(3),'go','MarkerSize',8,'MarkerFaceColor','g','DisplayName','Mars at Departure')

% Mark final positions at end of TOF
plot3(X1_TOF(end,1),X1_TOF(end,2),X1_TOF(end,3),'rs','MarkerSize',8,'MarkerFaceColor','r','DisplayName','Earth at Arrival')
plot3(X2_TOF(end,1),X2_TOF(end,2),X2_TOF(end,3),'gs','MarkerSize',8,'MarkerFaceColor','g','DisplayName','Mars at Arrival')

% Add Sun at origin
plot3(0, 0, 0,'o', 'MarkerSize', 12, 'MarkerFaceColor', '#FFA500', 'DisplayName', 'Sun');

title('Hohmann Transfer to Mars')
xlabel('Distance (km)')
ylabel('Distance (km)')
legend('show')
axis equal
grid on


end 

%% Functions --------------------------------------------------------------

% Calculate delta V
function deltaV = deltaV_calc(r1,r2,a1,a2,at,e1,e2,et,f1,f2,I1,I2,alpha,beta,mu)  % works only for hohmann
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
if I1 == 0 && I2 == 0  
    dv1 = sqrt(v1^2 + vt1^2 - 2*v1*vt1*cos(dg1));  % no inclination change
else
    dv1 = sqrt(v1^2 + vt1^2 - 2*v1*vt1*(cos(alpha))); 
end
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
if I1 == 0 && I2 == 0
    dv2 = sqrt(v2^2 + vt2^2 - 2*v2*vt2*cos(dg2));  % no inclination change
else
    dv2 = sqrt(v2^2 + vt2^2 - 2*v2*vt2*cos(beta)); 
end
deltaV = dv1 + dv2;
if I1 ~= I2
fprintf('\n For Alpha = %.4f, Total Delta V required for Transfer = %.4f km/s',rad2deg(alpha),deltaV)
else
fprintf('\nTotal Delta V required for Transfer = %.4f km/s',deltaV)
fprintf('\nDelta V from Orbit 1 to Transfer orbit is %.4f Km/s', dv1); 
fprintf('\nDelta V from Transfer Orbit to Orbit 2 is %.4f Km/s', dv2); 
end
end
%--------------------------------------------------------------------------

% Calculates r and v vectors to ECI frame
function [r_vec, v_vec] = orbparm_to_ECI(a,e,f,mu,inc,raan,aop)
    P = a*(1-e^2); 
    r = P/(1+e*cos(f));
    % perifocal cords (EPH)
    r_vec = [r*cos(f); r*sin(f);0]; 
    v_vec = [-sqrt(mu/P)*sin(f); sqrt(mu/P)*(e+cos(f));0]; 
    % for plane change --> Need DCM's to transform from Perifocal to ECI 
    % Rotation matrices
    R1 = [cos(raan), -sin(raan), 0; sin(raan),  cos(raan), 0; 0,0,1];
    R2 = [1, 0,0;0, cos(inc), -sin(inc); 0, sin(inc), cos(inc)]; 
    R3 = [cos(aop), -sin(aop), 0; sin(aop),  cos(aop), 0; 0,0,1]; 
    % Complete transformation matrix from perifocal to ECI
    DCM = R1 * R2 * R3;
    % Transform to ECI
    r_vec = DCM * r_vec;
    v_vec = DCM * v_vec;
end
%--------------------------------------------------------------------------

% Initiates State Space eqns
function Xdot = statespace(~,mu,X)
    r_vec = X(1:3);
    v_vec = X(4:6);
    r = norm(r_vec);
    a_vec = -mu / r^3 * r_vec;
    Xdot = [v_vec; a_vec];
end
%--------------------------------------------------------------------------

% Newton Raphson
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