from astropy.time import Time
from astropy.time import TimeDelta
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body

t1 = Time("2014-09-22 23:22")
td = TimeDelta(50.0, format='sec')
td2 = TimeDelta(50.0, format='jd')

t2 = t1+td
t3 = t1+td2

solar_system_ephemeris.set('de432s')


print(get_body_barycentric('earth', t1))
print(get_body_barycentric('earth', t2))
print(get_body_barycentric('earth', t3))
