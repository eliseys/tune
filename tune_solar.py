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

from astropy.coordinates import ICRS
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
import astropy.units as u
from astropy.time import Time, TimeDelta
from datetime import timedelta, time, datetime

import pytz
from timezonefinder import TimezoneFinder

from sh import gphoto2 as gp

from exif import Image as exif_data

from skyfield.api import load, Topos, wgs84

from astropy.utils.iers import conf
conf.auto_max_age = None
conf.auto_download = True

from skimage.draw import disk
from skimage.io import imsave, imread

from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks

import matplotlib.pyplot as plt





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



#def solve_field(image_name, configuration):
#
#    s = configuration 
#    
#    #scale = (s.pixel_size/s.focal_lenght) * (180 * 60 * 60/np.pi)
#    #scale_low = scale * (1 - 0.1)
#    #scale_high = scale * (1 + 0.1)
#
#    scale_low = s.pixel_scale * (1.0 - 0.05)
#    scale_high = s.pixel_scale * (1.0 + 0.05)
#
#    
#    image_name_base, image_name_extention = os.path.splitext(image_name)
#
#    if os.path.exists(image_name_base + ".new"):
#        os.remove(image_name_base + ".new")
#
#    
#    if s.star_extraction_method == "sex":
#        
#        os.system("solve-field --no-plots --overwrite --use-source-extractor --source-extractor-path {} --source-extractor-config {} --x-column X_IMAGE --y-column Y_IMAGE --sort-column MAG_AUTO --sort-ascending --scale-units arcsecperpix --scale-low {} --scale-high {} ".format(s.path_to_sextractor, s.sextractor_config, scale_low, scale_high) + image_name + " > /dev/null 2>&1")
#
#        
#    elif s.star_extraction_method == "image2xy":
#        
#        os.system("solve-field --no-plots --overwrite --scale-units arcsecperpix --scale-low {} --scale-high {} ".format(scale_low, scale_high) + image_name + " > /dev/null 2>&1")
#    
#
#
#    with fits.open(image_name_base + ".new") as f:
#        w = WCS(f[0].header)
#
#    return w



def test_images():

    fig, ax = plt.subplots(figsize=(10, 6))


    #img = np.zeros((200, 300, 3), dtype=np.double)

    img = np.zeros((200, 300, 3), dtype=np.double)

    rr, cc = disk((120, 150), 50, shape=img.shape)

    img[rr, cc, :] = (1, 1, 0)

    
    imsave("image_3.tiff", img)
    ax.imshow(img)
    #ax.set_title('No anti-aliasing')
    #ax.axis('off')
    plt.show()


    return 0




def circle_detection(image_filename, s):

    ts = s.ts
    observer_location = s.observer_location()
    location, location_message = s.location()

    sun = s.sun()
    sun_r = s.sun_radius
    
    apparent_sun = observer_location.at(t).observe(sun).apparent()

    _, _, distance_sun = apparent_sun.altaz()

    #print(distance_sun.km)
    
    
    sun_angular_radius = np.arctan(sun_r/distance_sun.km)
    
    sun_pixel_radius = sun_angular_radius/s.pixel_scale

    ######
    sun_pixel_radius = 50
    ######


    image = imread(image_filename)

    image = image[:, :, 1]

    edges = canny(image, sigma=3)

    
    hough_radii = np.arange(sun_pixel_radius - 3, sun_pixel_radius + 3, 1.0)

    hough_res = hough_circle(edges, hough_radii)
    
    # Select the most prominent circle
    accums, cx, cy, radius = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
    
    
    #print(radii)
    #print(cx, cy)
    #
    ## Draw them
    #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 15))
    ##image = color.gray2rgb(image)
    #
    #
    #for center_y, center_x, radius in zip(cy, cx, radii):
    #    circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
    #    image[circy, circx] = image.max()
    #    
    #    
    #    ax.imshow(image)
    #    
    #    #ax.imshow(edges, cmap=plt.cm.gray)
    #    
    #    
    #    plt.show()

    #print(cx[0], cy[0], radius[0])


    #return x, y # pixel coordinates of the sun center
    return cx[0], cy[0], radius[0]




def shift(images, s):

    image1, image2, image3 = images

    # center of the Sun pixel coordinates
    x1, y1, _ = circle_detection(image1, s)
    x2, y2, _ = circle_detection(image2, s)
    x3, y3, _ = circle_detection(image3, s)

    # y = a*x + b
    a = (y1 - y2)/(x1 - x2)
    b = (x1*y2 - x2*y1)/(x1 - x2)

    # shift, pixels
    l = (y3 - a*x3 - b)/np.sqrt(a**2 + 1)

    # shift, arcseconds
    shift_arcseconds = l * s.pixel_scale

    
    return shift_arcseconds 
    
    


def axis_vector(images, s):

    ts = s.ts
    observer_location = s.observer_location()
    location, location_message = s.location()

    sun = s.sun()

    image_1, image_2, image_3 = images

    #t1 = s.exif_time(image_2)
    #t2 = s.exif_time(image_3)
    
    t1 = ts.utc(2024, 3, 8, 12, 35, 37.5)
    t2 = ts.utc(2024, 3, 8, 12, 40, 37.5)

    apparent_sun_1 = observer_location.at(t1).observe(sun).apparent()
    apparent_sun_2 = observer_location.at(t2).observe(sun).apparent()

    alt_1, az_1, distance_1 = apparent_sun_1.altaz()
    alt_2, az_2, distance_2 = apparent_sun_2.altaz()


    a1 = altaz2dec(alt_1.radians, az_1.radians)
    a2 = altaz2dec(alt_2.radians, az_2.radians)


    #print("a1", a1, type(a1), len(a1))
    #print("a2", a2, type(a2), len(a2))

    
    sun_vector = (a1 + a2)

    sun_vector = sun_vector/np.linalg.norm(sun_vector)
    
    pole = pole_vector(t2.to_astropy(), s)


    sun_vector_projection = np.dot(sun_vector, pole)


    aux_axis = sun_vector - pole * sun_vector_projection


    shift_arcseconds = shift(images, s)

    #shift_arcseconds = -150.0


    # angle between firt and second solar position
    c = np.arccos(np.dot(a1, a2))

    shift_radians = shift_arcseconds * (np.pi/(180*60*60))

    #shift_angle = np.arccos((np.cos(shift_radians) - np.cos(c)**2)/(np.sin(c)**2))

    shift_angle = np.arctan(shift_radians/c)

    
    #print("shift angle, degrees", shift_angle*180.0/np.pi)

    
    axis = Rotation.from_rotvec(shift_angle * aux_axis).apply(pole)

    #print("pole", pole)

    #print("angle pole axis", np.arccos(np.dot(pole, axis))*180.0/np.pi)

    
    return axis





def pole_vector(time, s):

    location, _ = s.location()
    
    if location.geodetic.lat.degree >= 0:    
        pole = SkyCoord(0.0, 90.0, frame='cirs', unit='deg', obstime=time, location=location)
    elif location.geodetic.lat.degree < 0:
        pole = SkyCoord(0.0, -90.0, frame='cirs', unit='deg', obstime=time, location=location)

    altaz_frame = AltAz(obstime=time, location=location)
    
    pole_altaz = pole.transform_to(altaz_frame)

    pole = altaz2dec(pole_altaz.alt.radian, pole_altaz.az.radian)
    
    return pole 




    

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
    R_fixed = s.R_fixed
    L_fixed = s.L_fixed

    
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


    if R_fixed == False and L_fixed == False:
        condition_1 = (abs(res_fr.x) <= abs(res_fl.x))
        condition_2 = (abs(res_fr.x) > abs(res_fl.x))
    else:
        condition_1 = (not R_fixed)
        condition_2 = (not L_fixed)

    
    if condition_1:

        if res_r.x >= 0:
            print("R {:.3f} {}".format(abs(res_r.x / q_R), anticlockwise))
        elif res_r.x < 0:
            print("R {:.3f} {}".format(abs(res_r.x / q_R), clockwise))

        if res_fr.x >= 0:
            print("F {:.3f} {}".format(abs(res_fr.x / q_F), anticlockwise))
        elif res_fr.x < 0:
            print("F {:.3f} {}".format(abs(res_fr.x / q_F), clockwise))

    elif condition_2:

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
    print("Tangential misalignment (!)")
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

        self.L_fixed = None
        self.R_fixed = None
        
        self.moon_radius = 1737.4 
        self.sun_radius = 696000.0

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



    @property
    def R_fixed(self):
        return self._R_fixed

    @R_fixed.setter
    def R_fixed(self, value):
        self._R_fixed = value



    @property
    def L_fixed(self):
        return self._L_fixed

    @L_fixed.setter
    def L_fixed(self, value):
        self._L_fixed = value
        
        
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


    #def manual_location(self):

    #    try:
    #        location = EarthLocation.from_geodetic(lat=self.latitude*u.deg, lon=self.longitude*u.deg, height=self.altitude*u.m)
    #        message = "DEFAULT LOCATION"

    #    except:
    #        location = None
    #        message = "DEFAULT LOCATION FAILED"

    #    
    #    return location, message



    def manual_location(self):

        #print("latitude", self.latitude)
        
        try:
            location = EarthLocation.from_geodetic(lat=self.latitude*u.deg, lon=self.longitude*u.deg, height=self.altitude*u.m)

            #location = wgs84.latlon(self.latitude, self.longitude, elevation_m = self.altitude)

            message = "DEFAULT LOCATION"


            
        except:
            location = None
            message = "DEFAULT LOCATION FAILED"

        
        return location, message




    

    def location(self):

        try:
            image_file_name = self.image_file_name_list[-1]
        except:
            image_file_name = None

            
        location = None

        if type(location) == type(None) and self.gpsd:
            location, message = self.gpsd_location()

        elif type(location) == type(None):
            try:
                location, message = self.exif_location(image_file_name)
            except:
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


    
    ephem = load("de440s.bsp")
    ts = load.timescale()

    #earth, moon, sun = ephem['earth'], ephem['moon'], ephem['sun']

    def earth(self):
        return self.ephem['earth']


    def moon(self):
        return self.ephem['moon']


    def sun(self):
        return self.ephem['sun']


    
    def observer_location(self):
        

        #ephem = load("de440s.bsp")
        #earth = ephem['earth']

        #location = wgs84.latlon(self.latitude, self.longitude, elevation_m = self.altitude)

        #return self.earth() + self.location()[0]

        return self.earth() + wgs84.latlon(self.latitude, self.longitude, elevation_m = self.altitude)

    #utc = timezone(s.utc_timezone)

    
    def local_timezone(self):

        tf = TimezoneFinder()
        return tf.timezone_at(lng=self.longitude, lat=self.latitude)



    

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

    parser = argparse.ArgumentParser(description='Tune solar --- daytime polar axis alignment tool')
    
    #parser.add_argument('-n', choices=[2, 3], default=2, dest='N', type=int, nargs=1, help='Number of shots: 2 for polar region, 3 for arbitrary sky region')

    parser.add_argument('-t', default=300, dest='t', type=int, nargs=1, help='Time delay between shots, seconds')

    parser.add_argument('-f', dest='files', action="append", nargs='+', type=str, help='List the names of the image files to be processed')

    args = parser.parse_args()


    s = configuration()

    
    with open('tune_solar.json', 'r') as f:
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

    s.L_fixed = np.array(data["tripod"]["L_fixed"])
    s.R_fixed = np.array(data["tripod"]["R_fixed"])
    
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


        images = []
        system_times = []
        for i in range(3):
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

    



    ts = s.ts
    t = ts.utc(2024, 3, 8, 12, 40, 37.5)



    images = ["image_1.tiff", "image_2.tiff", "image_3.tiff"]


    axis = axis_vector(images, s)
    pole = pole_vector(t.to_astropy(), s)

    
    misalignment(axis, pole)

    tripod_adjucements(axis, pole, s)
        
    
    #list_dir = os.listdir('.')

    #image_name_bases = []
    #for image_name in images:
    #    image_name_base, _ = os.path.splitext(image_name)
    #    image_name_bases.append(image_name_base)
    #    image_name_bases.append(image_name_base+"-indx")

    #
    #for file_name in list_dir:
    #    file_name_base, file_name_extention = os.path.splitext(file_name)

    #    if file_name_base in image_name_bases:
    #       shutil.move(file_name, s.imagedir + "/" + file_name)
    #        


    

