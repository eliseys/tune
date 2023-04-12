#!/usr/bin/python3

import sys
import shutil
import os
import argparse
import errno
import json

import numpy as np

from scipy.spatial.transform import Rotation
from scipy.optimize import minimize_scalar, minimize

from astropy.io import fits

from astropy.wcs import WCS

from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
import astropy.units as u
from astropy.time import Time, TimeDelta
from datetime import timedelta, time, datetime

import pytz
from timezonefinder import TimezoneFinder

from sh import gphoto2 as gp

from exif import Image as exif_data

#from astropy.utils.iers import conf
#conf.auto_max_age = None




def altaz2dec(alt, az):
    x = np.cos( - az) * np.sin(np.pi/2 - alt)
    y = np.sin( - az) * np.sin(np.pi/2 - alt)
    z = np.cos(np.pi/2 - alt)
    
    return np.array([x, y, z])


def xy_projector(a):
    P_xy = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    a_xy = np.matmul(P_xy, a)
    return a_xy



def solve_field(image_name, configuration):

    s = configuration 
    
    #scale = (s.pixel_size/s.focal_lenght) * (180 * 60 * 60/np.pi)
    #scale_low = scale * (1 - 0.1)
    #scale_high = scale * (1 + 0.1)

    scale_low = s.pixel_scale * (1.0 - 0.05)
    scale_high = s.pixel_scale * (1.0 + 0.05)

    
    image_name_base, image_name_extention = os.path.splitext(image_name)

    if os.path.exists(image_name_base + ".new"):
        os.remove(image_name_base + ".new")

    
    if s.star_extraction_method == "sex":
        
        os.system("solve-field --no-plots --overwrite --use-source-extractor --source-extractor-path {} --source-extractor-config {} --x-column X_IMAGE --y-column Y_IMAGE --sort-column MAG_AUTO --sort-ascending --scale-units arcsecperpix --scale-low {} --scale-high {} ".format(s.path_to_sextractor, s.sextractor_config, scale_low, scale_high) + image_name + " > /dev/null 2>&1")

        
    elif s.star_extraction_method == "image2xy":
        
        os.system("solve-field --no-plots --overwrite --scale-units arcsecperpix --scale-low {} --scale-high {} ".format(scale_low, scale_high) + image_name + " > /dev/null 2>&1")
    


    with fits.open(image_name_base + ".new") as f:
        w = WCS(f[0].header)

    return w



def axis_vector_3(times, ws, location):

    t1, t2, t3 = times
    w1, w2, w3 = ws
    
    x = w1.pixel_shape[0]/2
    y = w1.pixel_shape[1]/2

    a1_sky = w1.pixel_to_world(x, y)
    a2_sky = w2.pixel_to_world(x, y)
    a3_sky = w3.pixel_to_world(x, y)

    altaz_frame_1 = AltAz(obstime=t1, location=location)
    altaz_frame_2 = AltAz(obstime=t2, location=location)
    altaz_frame_3 = AltAz(obstime=t3, location=location)

    a1_altaz = a1_sky.transform_to(altaz_frame_1)
    a2_altaz = a2_sky.transform_to(altaz_frame_2)
    a3_altaz = a3_sky.transform_to(altaz_frame_3)

    a1 = altaz2dec(a1_altaz.alt.radian, a1_altaz.az.radian)
    a2 = altaz2dec(a2_altaz.alt.radian, a2_altaz.az.radian)
    a3 = altaz2dec(a3_altaz.alt.radian, a3_altaz.az.radian)

 
    a12 = a2 - a1
    a23 = a3 - a2


    axis = np.cross(a12, a23)

    axis = axis/np.linalg.norm(axis)

    return axis


def pole_vector_3(times, location):

    _, time, _ = times

    if location.geodetic.lat.degree >= 0:    
        pole = SkyCoord(0.0, 90.0, frame='cirs', unit='deg', obstime=time, location=location)
    elif location.geodetic.lat.degree < 0:
        pole = SkyCoord(0.0, -90.0, frame='cirs', unit='deg', obstime=time, location=location)

        
    altaz_frame = AltAz(obstime=time, location=location)
    
    pole_altaz = pole.transform_to(altaz_frame)

    pole = altaz2dec(pole_altaz.alt.radian, pole_altaz.az.radian)
    
    return pole 


def pixel_sky_separation(xy, w1, w2):

    x, y = xy
    
    sky_1 = w1.pixel_to_world(x, y)
    sky_2 = w2.pixel_to_world(x, y)
    sep = sky_1.separation(sky_2).arcsecond

    return sep




def axis_image_coordinates_wcs(ws):


    w1, w2 = ws
    
    x, y = w1.array_shape

    res = minimize(pixel_sky_separation, [x/2, y/2], args=(w1, w2), method='Nelder-Mead', tol=1e-5)

    x, y, = res.x

    return x, y
        


def pole_vector_2(times, location):

    time, _ = times
    
    if location.geodetic.lat.degree >= 0:    
        pole = SkyCoord(0.0, 90.0, frame='cirs', unit='deg', obstime=time, location=location)
    elif location.geodetic.lat.degree < 0:
        pole = SkyCoord(0.0, -90.0, frame='cirs', unit='deg', obstime=time, location=location)

    altaz_frame = AltAz(obstime=time, location=location)
    
    pole_altaz = pole.transform_to(altaz_frame)

    pole = altaz2dec(pole_altaz.alt.radian, pole_altaz.az.radian)
    
    return pole 



def axis_vector_2(times, ws, location):

    t1, t2 = times
    w1, w2 = ws
    
    x, y = axis_image_coordinates_wcs(ws)

    axis_sky = w1.pixel_to_world(x, y)
    
    altaz_frame = AltAz(obstime=t1, location=location)
    
    axis_altaz = axis_sky.transform_to(altaz_frame)

    axis = altaz2dec(axis_altaz.alt.radian, axis_altaz.az.radian)


    return axis

    

def tripod_parameters(axis, s):
    
    z = np.array([0,0,1])
    x = np.array([1,0,0])

    L = s.L
    F = s.F
    R = s.R
    
    thread_pitch = s.thread_pitch

    l = (R - F)/np.linalg.norm(R - F)
    r = (F - L)/np.linalg.norm(F - L)
    f = (L - R)/np.linalg.norm(L - R)

    axis_xy = xy_projector(axis)
    axis_xy = axis_xy/np.linalg.norm(axis_xy)

    u = np.sign(np.cross(x, axis_xy)) * np.arccos(np.dot(x, axis_xy))

    l = Rotation.from_rotvec(u * z).apply(l)
    r = Rotation.from_rotvec(u * z).apply(r)
    f = Rotation.from_rotvec(u * z).apply(f)

    alpha_R = np.arccos(np.dot(f, -l))
    alpha_L = np.arccos(np.dot(f, -r))
    
    q_L = thread_pitch/np.linalg.norm(L - R) * np.sin(alpha_R)
    q_R = thread_pitch/np.linalg.norm(L - R) * np.sin(alpha_L)
    q_F = thread_pitch/F[0]

    
    return r, f, l, q_R, q_F, q_L



def tripod_adjucements(axis, pole, s):


    axis_xy = xy_projector(axis)
    pole_xy = xy_projector(pole)

    axis_xy = axis_xy/np.linalg.norm(axis_xy)
    pole_xy = pole_xy/np.linalg.norm(pole_xy)
    
    delta_az = np.sign(np.cross(pole_xy, axis_xy)[2])*np.arccos(np.dot(axis_xy, pole_xy))
    delta_alt = np.arccos(np.dot(pole, pole_xy)) - np.arccos(np.dot(axis, axis_xy))

    separation = np.arccos(np.dot(axis, pole))


    clockwise = "\u21bb"
    anticlockwise = "\u21ba"


    r, f, l, q_R, q_F, q_L = tripod_parameters(axis, s)

    def obj_func_r(u, axis, pole):
        a = Rotation.from_rotvec(u * r).apply(axis)
        return -(a[0]*pole[0] + a[1]*pole[1])/(np.linalg.norm(np.array([a[0], a[1]]))*np.linalg.norm(np.array([pole[0], pole[1]])))
    
    
    def obj_func_l(u, axis, pole):
        a = Rotation.from_rotvec(u * l).apply(axis)
        return -(a[0]*pole[0] + a[1]*pole[1])/(np.linalg.norm(np.array([a[0], a[1]]))*np.linalg.norm(np.array([pole[0], pole[1]])))

    
    def obj_func_f(u, axis, pole):
        a = Rotation.from_rotvec(u * f).apply(axis)
        return -np.dot(a, pole)

    
    res_r = minimize_scalar(obj_func_r, bounds=(-0.5,0.5), args=(axis, pole), method='bounded', options={'xatol': 1e-12})
    ax_r = Rotation.from_rotvec(res_r.x * r).apply(axis)
    
    res_l = minimize_scalar(obj_func_l, bounds=(-0.5,0.5), args=(axis, pole), method='bounded', options={'xatol': 1e-12})
    ax_l = Rotation.from_rotvec(res_l.x * l).apply(axis)
    
    res_fr = minimize_scalar(obj_func_f, bounds=(-0.5,0.5), args=(ax_r, pole), method='bounded', options={'xatol': 1e-12})
    res_fl = minimize_scalar(obj_func_f, bounds=(-0.5,0.5), args=(ax_l, pole), method='bounded', options={'xatol': 1e-12})

    
    if abs(res_fr.x) <= abs(res_fl.x):

        if res_r.x >= 0:
            print("R {:.3f} {}".format(abs(res_r.x / q_R), anticlockwise))
        elif res_r.x < 0:
            print("R {:.3f} {}".format(abs(res_r.x / q_R), clockwise))

        if res_fr.x >= 0:
            print("F {:.3f} {}".format(abs(res_fr.x / q_F), anticlockwise))
        elif res_fr.x < 0:
            print("F {:.3f} {}".format(abs(res_fr.x / q_F), clockwise))

    elif abs(res_fr.x) > abs(res_fl.x):

        if res_l.x >= 0:
            print("L {:.3f} {}".format(abs(res_l.x / q_L), anticlockwise))
        elif res_l.x < 0:
            print("L {:.3f} {}".format(abs(res_l.x / q_L), clockwise))        

        if res_fl.x >= 0:
            print("F {:.3f} {}".format(abs(res_fl.x / q_F), anticlockwise))
        elif res_fl.x < 0:
            print("F {:.3f} {}".format(abs(res_fl.x / q_F), clockwise))

    print('\n')

    
    return 0



def misalignment(axis, pole):
    
    axis_xy = xy_projector(axis)
    pole_xy = xy_projector(pole)

    axis_xy = axis_xy/np.linalg.norm(axis_xy)
    pole_xy = pole_xy/np.linalg.norm(pole_xy)
    
    delta_az = np.sign(np.cross(pole_xy, axis_xy)[2])*np.arccos(np.dot(axis_xy, pole_xy))
    delta_alt = np.arccos(np.dot(pole, pole_xy)) - np.arccos(np.dot(axis, axis_xy))

    separation = np.arccos(np.dot(axis, pole))


    left_arrow = "\u2190"
    right_arrow = "\u2192"
    upwards_arrow = "\u2191"
    downwards_arrow = "\u2193"
    

    eps = (np.pi/(180.0 * 60 * 60)) * 0.05
    
    if delta_az > eps:
        az_arrow = right_arrow
    elif delta_az < - eps:
        az_arrow = left_arrow
    elif abs(delta_az) < eps:
        az_arrow = " "
        
    if delta_alt > eps:
        alt_arrow = upwards_arrow
    elif delta_alt < - eps:
        alt_arrow = downwards_arrow
    elif abs(delta_alt) < eps:
        alt_arrow = " "

    print("\n")
    print("\u0394    {}".format(Angle(separation, u.radian).to_string(unit=u.degree, sep=('\u00b0 ', '\u2032 ', '\u2033'), precision=1)))
    print("\u0394h  {} {}".format(Angle(delta_alt, u.radian).to_string(unit=u.degree, sep=('\u00b0 ', '\u2032 ', '\u2033'), precision=1, alwayssign=True), alt_arrow))
    print("\u0394\u03c6  {} {}".format(Angle(delta_az, u.radian).to_string(unit=u.degree, sep=('\u00b0 ', '\u2032 ', '\u2033'), precision=1, alwayssign=True), az_arrow))
    print('\n')


    return 0


class configuration:

    def __init__(self):
        self.image_file_name_list = []
        self._imagedir = None
        self._pixel_size = None 
        self._focal_lenght = None

        self.latitude = None 
        self.longitude = None 
        self.altitude = None

        self._thread_pitch = None
        self._L = None 
        self._F = None
        self._R = None

        self.star_extraction_method = None

        self.path_to_sextractor = None

        self.sextractor_config = None

        self.gpsd = None

        

    def add_image(self, image_file_name):
        self.image_file_name_list.append(image_file_name)


    @property
    def imagedir(self):
        return self._imagedir

    @imagedir.setter
    def imagedir(self, value):
        self._imagedir = value

        
        
    @property
    def pixel_size(self):
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, value):
        self._pixel_size = value



    @property
    def focal_lenght(self):
        return self._focal_lenght

    @focal_lenght.setter
    def focal_lenght(self, value):
        self._focal_lenght = value



    @property
    def pixel_scale(self):
        return (self._pixel_size/self._focal_lenght)*(180*60*60/np.pi)



    @property
    def thread_pitch(self):
        return self._thread_pitch

    @thread_pitch.setter
    def thread_pitch(self, value):
        self._thread_pitch = value


    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, value):
        self._L = value


    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, value):
        self._F = value

    
    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        self._R = value

        
        
    def gpsd_location(self):
        
        try:
            gpsd.connect()
            packet = gpsd.get_current()
            latitude = packet.position()[0]
            longitude = packet.position()[1]
            altitude = packet.altitude()
            location = EarthLocation.from_geodetic(lat=latitude*u.deg, lon=longitude*u.deg, height=altitude*u.m)
            message = "GPSD LOCATION"
        except:
            location = None
            message = ""

        return location, message

    
    def exif_location(self, image_file_name):
        
        with open(image_file_name, 'rb') as image:
            exif = exif_data(image)
            
        if exif.has_exif:
            pass
        else:
            #print("NO EXIF {}".format(image_file_name))
            message = ""
            location = None
            return location, message 
        
        
        try:
            altitude = exif.gps_altitude
            
            lat_d, lat_m, lat_s = exif.gps_latitude
            lat_ref = exif.gps_latitude_ref 
            
            lon_d, lon_m, lon_s = exif.gps_longitude
            lon_ref = exif.gps_longitude_ref 
            
            latitude = Angle("{}d{}m{}".format(int(lat_d), lat_m, lat_ref))
            longitude = Angle("{}d{}m{}".format(int(lon_d), lon_m, lon_ref))
            
            location = EarthLocation.from_geodetic(lat=latitude, lon=longitude, height=altitude*u.m)
            message = "EXIF LOCATION"
        except:
            message = ""
            location = None
                
        
        return location, message


    def manual_location(self):

        try:
            location = EarthLocation.from_geodetic(lat=self.latitude*u.deg, lon=self.longitude*u.deg, height=self.altitude*u.m)
            message = "DEFAULT LOCATION"

        except:
            location = None
            message = "DEFAULT LOCATION FAILED"

        
        return location, message


    def location(self):

        image_file_name = self.image_file_name_list[-1]
        
        location = None

        if type(location) == type(None) and self.gpsd:
            location, message = self.gpsd_location()

        if type(location) == type(None):
            location, message = self.exif_location(image_file_name)
            
        if type(location) == type(None):
            location, message = self.manual_location()
            
            
        return location, message

    

    def exif_time(self, image_file_name):

        with open(image_file_name, 'rb') as image:
            exif = exif_data(image)

        try:    
            t = exif.datetime_original.split()
        except:
            t = None
        
        
        if t == None:
            utc_time = None
            pass
        
        else:
            
            t = t[0].replace(':','-') + 'T' + t[1]
            
            t = datetime.strptime(t, "%Y-%m-%dT%H:%M:%S")
            
            tf = TimezoneFinder()

            l, _ = self.location()
            
            local_timezone = tf.timezone_at(lng=l.to_geodetic().lon.deg, lat=l.to_geodetic().lat.deg)
            local = pytz.timezone(local_timezone)
            utc = pytz.UTC
            
            t.astimezone(local)
            utc_time = t.astimezone(utc)
            
        return utc_time


    def time_to_utc(self, times):

        tf = TimezoneFinder()

        l, _ = self.location()
        
        local_timezone = tf.timezone_at(lng=l.to_geodetic().lon.deg, lat=l.to_geodetic().lat.deg)
        local = pytz.timezone(local_timezone)
        utc = pytz.UTC

        utc_times = []
        for t in times:
            t.astimezone(local)
            utc_times.append(t.astimezone(utc))

        return utc_times


        
    
    def exif_time_list(self):
        time_list = []
        
        for image_file_name in self.image_file_name_list: 
            time_list.append(self.exif_time(image_file_name))


        return time_list


    

    

def camera_setup():
    gp("--set-config", "imageformat=0") # Large Fine JPEG
    gp("--set-config", "capturetarget=1") # write first to memory card




def capture_image():

    input()

    list_dir_before = os.listdir('.')

    try:
        print("Capturing image ... ", end="\r")
        time = datetime.now()
        s = gp("--capture-image-and-download")
    except:
        print("Capturing image failed")

    list_dir_after = os.listdir('.')

    new_files = list(set(list_dir_after) - set(list_dir_before))

    if len(new_files) == 1:
        image_file_name = new_files[0]
        sys.stdout.write("\x1b[2K\r")
        print("File {} downloaded".format(image_file_name), end="")
    elif len(new_files) > 1:
        print("Multiple files has been added: ", new_files)
        image_file_name = None 
    elif len(new_files) == 0:
        print("No files has been added")
        image_file_name = None 
    
    return image_file_name, time

    





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Tune --- polar axis alignment tool')
    
    parser.add_argument('-n', choices=[2, 3], default=2, dest='N', type=int, nargs=1, help='Number of shots: 2 for polar region, 3 for arbitrary sky region')

    parser.add_argument('-f', dest='files', action="append", nargs='+', type=str, help='List the names of the image files to be processed')

    args = parser.parse_args()


    s = configuration()

    
    with open('tune.json', 'r') as f:
        data = json.load(f)

    data = data["RPi"]

        
    s.imagedir = data["imagedir"]

    s.pixel_size = data["pixel_size"]
    s.focal_lenght = data["focal_lenght"]

    s.latitude = data["location"]["latitude"]
    s.longitude = data["location"]["longitude"]
    s.altitude = data["location"]["altitude"]
    
    s.gpsd = data["gpsd"]
    
    s.thread_pitch = data["tripod"]["thread_pitch"]
    s.L = np.array(data["tripod"]["L"])
    s.F = np.array(data["tripod"]["F"])
    s.R = np.array(data["tripod"]["R"])

    s.star_extraction_method = data["star_extraction_method"] 

    s.path_to_sextractor = data["path_to_sextractor"]

    s.sextractor_config = data["sextractor_config"]


    
    if not os.path.exists(s.imagedir):
        os.makedirs(s.imagedir)


    
    images = None

    if args.files == None:
        pass
    elif type(args.files[0]) == list:
        images = args.files[0]
        for image in images:
            if os.path.exists(image):
                pass
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), image)

        N = len(images)
    else:
        parser.print_help()

        
        
    if images == None:

        try:
            camera_setup()
            print("Camera ready")
        except:
            print("Camera setup failed")


        
        N = args.N
        if type(N) == list:
            N = N[0]
        else:
            pass

        
        images = []
        system_times = []
        for i in range(N):
            image_file_name, system_time = capture_image()
            images.append(image_file_name)
            system_times.append(system_time)


    for image_name in images:
        s.add_image(image_name)

    
    location, location_message = s.location()

    print('\n')
    print(location_message)

    
    try:
        times = s.exif_time_list()
    except:
        times = s.time_to_utc(system_times)
        print("SYSTEM TIME")

    
    ws = []
    for image in images:

        print("Solving {} ...".format(image), end='\r')
        
        try:
            w = solve_field(image, s)
            ws.append(w)
        except:
            sys.stdout.write("\x1b[2K\r")
            print("Solving {} failed".format(image))
            w = None
            ws.append(w)

        if os.path.exists(os.path.splitext(image)[0] + ".new"):
            sys.stdout.write("\x1b[2K\r")
            print("{} solved".format(image))
        else:
            print("No {} file".format(os.path.splitext(image)[0] + ".new"))
        
            
    print("\n")



    

    if N == 3:
        axis = axis_vector_3(times, ws, location)
        pole = pole_vector_3(times, location)
        
        misalignment(axis, pole)

        tripod_adjucements(axis, pole, s)

    elif N == 2:
        axis = axis_vector_2(times, ws, location)
        pole = pole_vector_2(times, location)
        
        misalignment(axis, pole)

        tripod_adjucements(axis, pole, s)

    elif N == 1:
        pass


    list_dir = os.listdir('.')

    image_name_bases = []
    for image_name in images:
        image_name_base, _ = os.path.splitext(image_name)
        image_name_bases.append(image_name_base)
        image_name_bases.append(image_name_base+"-indx")

    
    for file_name in list_dir:
        file_name_base, file_name_extention = os.path.splitext(file_name)

        if file_name_base in image_name_bases:
           shutil.move(file_name, s.imagedir + "/" + file_name)
            


    

