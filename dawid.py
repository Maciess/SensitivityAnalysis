import numpy as np


def otl(Rb1, Rb2, Rf, Rc1, Rc2, beta):

    Vb1 = 12 * Rb2 / (Rb1 + Rb2)
    
    bR9 = beta * (Rc2 + 9)
    
    w1 = 0.74
    w2 = 11.35
    
    s1 = (Vb1 + w1) * bR9 / (bR9 + Rf)
    s2 = w2 * Rf / (bR9 + Rf)
    s3 = w1 * Rf * bR9 / ((bR9 + Rf) * Rc1)
    
    return s1 + s2 + s3


def otl_normalized(xx):
    return otl(
        translate_interval(50, 150, xx[:,0]),
        translate_interval(25, 70, xx[:,1]),
        translate_interval(0.5, 3, xx[:,2]),
        translate_interval(1.2, 2.5, xx[:,3]),
        translate_interval(0.25, 1.2, xx[:,4]),
        translate_interval(50, 300, xx[:,5]),
    )
    
    
def translate_interval(a, b, xx):
    avg = (a + b) / 2
    scale = b - avg
    
    return xx * scale + avg
