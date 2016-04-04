# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 11:34:04 2015

@author: skylion
"""
from flask import Flask, Request, request
from StringIO import StringIO
import unittest

 
RESULT = False


class TestFileFail(unittest.TestCase):
 
    class FileObj(StringIO):            
         def close(self):
             print 'in file close'
             global RESULT
             RESULT = True
        
    class MyRequest(Request):
        def _get_file_stream(*args, **kwargs):
            return FileObj()
app = Flask("server_handler")
 
app.debug = True
app.request_class = MyRequest
 
@app.route("/upload", methods=['POST'])
def upload():
    f = request.files['file']
    print 'in upload handler'
    self.assertIsInstance(
    f.stream,
    FileObj,
    )
    # Note I've monkeypatched werkzeug.datastructures.FileStorage 
    # so it wont squash exceptions
    f.close()
    #f.stream.close()
    return 

        
 
import pytesseract
import urllib, cStringIO
from PIL import Image


import glob
import os
import random
import sys
import random
import math
import json
from collections import defaultdict

import cv2
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage.filters import rank_filter

 
import urllib2
import urllib
import httplib
from xml.etree import ElementTree as etree
 
 
 
class wolfram(object):
    
    def __init__(self, appid):
        self.appid = appid
        self.base_url = 'http://api.wolframalpha.com/v2/query?'
        self.headers = {'User-Agent':None}

    def _get_xml(self, ip):
        url_params = {'input':ip, 'appid':self.appid}
        data = urllib.urlencode(url_params)
        req = urllib2.Request(self.base_url, data, self.headers)
        xml = urllib2.urlopen(req).read()
        return xml

    def _xmlparser(self, xml):
        data_dics = {}
        tree = etree.fromstring(xml)
        #retrieving every tag with label 'plaintext'
        for e in tree.findall('pod'):
            for item in [ef for ef in list(e) if ef.tag=='subpod']:
                for it in [i for i in list(item) if i.tag=='plaintext']:
                    if it.tag=='plaintext':
                        data_dics[e.get('title')] = it.text
        return data_dics

    def search(self, ip):
        xml = self._get_xml(ip)
        result_dics = self._xmlparser(xml)
        #return result_dics
        #print result_dics
        return result_dics['Result']


def getAnswers(outputs):
    my_wolf = wolfram('YAAR4P-R42A27KX7H')
    answers = [my_wolf.search('solve ' + output) for output in outputs]
    readEquationSheet()    
    print answers
    
def readEquation(input):    
    image = Image.open(input)    
    w, h = image.size    
    image.crop((0, 30, w, h-30))     
    pytesseract.pytesseract.tesseract_cmd = "tesseract"
    print pytesseract.image_to_string(image, lang="equ+eng") 

def readEquationSheet(input):
    image = Image.open(input)    
    input+= " math"
    pytesseract.pytesseract.tesseract_cmd = "tesseract"
    text = pytesseract.image_to_string(image, lang="equ+eng") 
    #print text    
    text = text.strip()    
    lines = text.split("\n")[2:-1]
    lines = lines[1:]    
    #print lines
    
    lines = [line for line in lines if line.strip() != ""]
    
    #print lines    
    
    operand_lines = lines[1::2]
    base_lines = lines[::2]
    bases = [line.split(" ") for line in base_lines]
    lines = ["".join(line.split()) for line in lines if line != ""]
    import re
    p=re.compile(r'[+$%X*-]')    
    operands = [p.split(line)[1:] for line in operand_lines]    
    #operands = [line.split("X")[1:] for line in operand_lines if 'X' in line]
    #operands = [line.split("//")[1:] for line in operand_lines]
    #operands = [line.split("%")[1:] for line in operand_lines]
   # operands = [line.split("-")[1:] for line in operand_lines]    
    #operands = [line for line in operands if line.strip() != ""]
    
    operators = [c for c in ''.join(operand_lines) if c in ['+','*','-', 'X', '%', '*']]
    output = []
    for base, operand, operator in zip(bases, operands, operators):
        for one_base, one_operand, one_operator in zip(base, operand, operator):            
            output.append(str(one_base) + " " +  str(one_operator) + str(one_operand))
    print output
    return output
    #print "Output: " + str(output)
    #print "NEW STUFF"

def downscale_image(im, max_dim=2048):
    """Shrink im until its longest dimension is <= max_dim.
    Returns new_image, scale (where scale <= 1).
    """
    a, b = im.size
    if max(a, b) <= max_dim:
        return 1.0, im

    scale = 1.0 * max_dim / max(a, b)
    new_im = im.resize((int(a * scale), int(b * scale)), Image.ANTIALIAS)
    return scale, new_im
    
    
def readSpreadSheet(input):
    orig_im = Image.open(input)
    scale, im = downscale_image(orig_im)

    edges = cv2.Canny(np.asarray(im), 100, 200)

    # TODO: dilate image _before_ finding a border. This is crazy sensitive!
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    borders = find_border_components(contours, edges)
    borders.sort(key=lambda (i, x1, y1, x2, y2): (x2 - x1) * (y2 - y1))
    
    border_contour = None
    if len(borders):
        border_contour = contours[borders[0][0]]
        edges = remove_border(border_contour, edges)

    edges = 255 * (edges > 0).astype(np.uint8)

    # Remove ~1px borders using a rank filter.
    maxed_rows = rank_filter(edges, -4, size=(1, 20))
    maxed_cols = rank_filter(edges, -4, size=(20, 1))
    debordered = np.minimum(np.minimum(edges, maxed_rows), maxed_cols)
    edges = debordered
    
    contours = find_components(edges)
    if len(contours) == 0:
        print '%s -> (no text!)' % input
        return
    return contours

def dilate(ary, N, iterations): 
    """Dilate using an NxN '+' sign shape. ary is np.uint8."""
    kernel = np.zeros((N,N), dtype=np.uint8)
    kernel[(N-1)/2,:] = 1
    dilated_image = cv2.dilate(ary / 255, kernel, iterations=iterations)

    kernel = np.zeros((N,N), dtype=np.uint8)
    kernel[:,(N-1)/2] = 1
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)
    return dilated_image


def props_for_contours(contours, ary):
    """Calculate bounding box & the number of set pixels for each contour."""
    c_info = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        c_im = np.zeros(ary.shape)
        cv2.drawContours(c_im, [c], 0, 255, -1)
        c_info.append({
            'x1': x,
            'y1': y,
            'x2': x + w - 1,
            'y2': y + h - 1,
            'sum': np.sum(ary * (c_im > 0))/255
        })
    return c_info


def union_crops(crop1, crop2):
    """Union two (x1, y1, x2, y2) rects."""
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)


def intersect_crops(crop1, crop2):
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return max(x11, x12), max(y11, y12), min(x21, x22), min(y21, y22)


def crop_area(crop):
    x1, y1, x2, y2 = crop
    return max(0, x2 - x1) * max(0, y2 - y1)


def find_border_components(contours, ary):
    borders = []
    area = ary.shape[0] * ary.shape[1]
    for i, c in enumerate(contours):
        x,y,w,h = cv2.boundingRect(c)
        if w * h > 0.5 * area:
            borders.append((i, x, y, x + w - 1, y + h - 1))
    return borders


def angle_from_right(deg):
    return min(deg % 90, 90 - (deg % 90))


def remove_border(contour, ary):
    """Remove everything outside a border contour."""
    # Use a rotated rectangle (should be a good approximation of a border).
    # If it's far from a right angle, it's probably two sides of a border and
    # we should use the bounding box instead.
    c_im = np.zeros(ary.shape)
    r = cv2.minAreaRect(contour)
    degs = r[2]
    if angle_from_right(degs) <= 10.0:
        box = cv2.cv.BoxPoints(r)
        box = np.int0(box)
        cv2.drawContours(c_im, [box], 0, 255, -1)
        cv2.drawContours(c_im, [box], 0, 0, 4)
    else:
        x1, y1, x2, y2 = cv2.boundingRect(contour)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 255, -1)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 0, 4)

    return np.minimum(c_im, ary)    

def find_components(edges, max_components=16):
    """Dilate the image until there are just a few connected components.
    Returns contours for these components."""
    # Perform increasingly aggressive dilation until there are just a few
    # connected components.
    count = 21
    dilation = 5
    n = 1
    while count > 16:
        n += 1
        dilated_image = dilate(edges, N=3, iterations=n)
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)
    #print dilation
    #Image.fromarray(edges).show()
    #Image.fromarray(255 * dilated_image).show()
    return contours

readEquation("/home/skylion/Pictures/simple_math.png")
readEquation("/home/skylion/Pictures/math_minute.png")
#math_minute = readSpreadSheet("/home/skylion/Pictures/math_minute.png")
readEquation("/home/skylion/Documents/hackathon/simple_math.crop.png")
math_minute = readEquationSheet("/home/skylion/Pictures/math_minute.png")

#print "HELLO"
math_minute2 = readEquationSheet("/home/skylion/Pictures/math_math2.png")

