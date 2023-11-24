import matplotlib.pyplot as plt 
import math
import numpy as np

E = 4000
Length = 220
mu = 0.2 # Ratio


def graphing():
    L = 1200
    n = 1200
    P = 400

    start_x = [120]
    x_train = [52, 228, 392, 568, 732, 908]
    p_train = [90, 90, 66.7, 66.7, 66.7, 66.7]

    A_y = []
    B_y = []

    for i in range(len(start_x)):
        moment = 0.0
        for j in range(len(x_train)):
            distance = x_train[j] + start_x[i]
            moment += (distance) * p_train[j]
        B_y.append(moment / 1200.0)
        A_y.append(sum(p_train) - B_y[i])

    print(A_y[0])
    print(B_y[0])
    
    SFDs= []
    BMDs= []
    for i in range(len(start_x)):
        SFD = [A_y[i]]
        BMD = [0]
        for j in range(1, 1201):
            SFD.append(SFD[j-1])
            if (j - start_x[i]) in x_train:
                indx = x_train.index(j-start_x[i])
                SFD[j] = SFD[j] - p_train[indx]
                BMD.append(BMD[j-1])
            else:
                BMD.append(BMD[j-1] + SFD[j])
        
        
        plt.xlim(0, 1201)                 
        graph, (plt1, plt2) = plt.subplots(1, 2)
        points = [start_x[i] + x for x in x_train]
        
        BMD = [-1 * ele for ele in BMD]
        
        x = range(0,1201)
        
        plt1.plot(x, SFD)
        plt1.plot(points, [SFD[x.index(p)] for p in points], 'ro', label='Highlighted Points')  # 'ro' means red circles
        plt1.set_title("SFD")
        
        
        
        plt2.plot(x, BMD)
        plt2.plot(points, [BMD[x.index(p)] for p in points], 'ro', label='Highlighted Points')  # 'ro' means red circles
        plt2.set_title("BMD")

        graph.tight_layout()
        plt.show()



def geography():
    
    
    #Setting up shear force,
    #I used max force for this one
    V = 300.2723333333333 # This value is obtained when  length is 1260mm
    M = 77199.71466666664 
    
    #Location of each cross section:
    y_bot, y_top, I, Qcent, Qglue, b, height = section_properties_left()
    
    
    
    #Stress part calculation
    bridge_shear = V * Qcent/ I / b
    glue_sheer = V * Qglue / I / b
    max_bridge_shear = np.amax(bridge_shear)
    max_glue_shear = np.amax(glue_sheer)
    compressions = M * y_top / I
    tensions = M * y_bot / I
    max_compression = np.amax(compressions)
    max_tension = np.amax(tensions)

    error_val = 0.5
    for top, bot in zip(y_top, y_bot):
        print(f"Top: {top}, Bot: {bot}")
        print(f"Ratio : {round(bot / top, 4)}")
        if bot / top > 5 + error_val:
            print("The botttom is too long or top is too short")
        elif bot / top < 5 - error_val:   
            print("The botttom is too short or top is too long")
        else:
            print("It is in good range")
        print()
    print("-----------------------------------")
    print(f"Max Bridge Shear: {round(max_bridge_shear , 4)} Max Glue Shear: {round(max_glue_shear, 4)} \nMax Compression: {round(max_compression, 4)} Max tension: {round(max_tension, 4)}")
    print("-----------------------------------")
    plate_buckling(y_top, height) 
    
#Take k t(thickness) and b to calculat the sigma_crit
def calculate_sigma(k, t, b):
    return k * math.pi ** 2 * E / (12 * (1 - mu**2)) * (t/b) ** 2

def plate_buckling(y_top, height):
    
    sigma_crit = []
    #Location of cross section
    x = [20, 100, 180, 280]
    #Location of diagphram
    a = [130, 300, 300, 130]




    #Case 1  k = 4
    k = 4
    t = 1.27 * 2 #thickness

    for i in range(1, len(x)):
        b = x[i] - x[i-1]
        sigma = calculate_sigma(k, t, b)
        sigma_crit.append(sigma)

    print("Sigma Crit for Case1: ")
    for sig in sigma_crit:
        print(round(sig, 4), end = " ")
    print()
    if not(all(sig >= 6 for sig in sigma_crit)):
        print("Case 1 Fails. The bridge is under thin plate buckling")
    print()   
    #Case 2 k = 0.425
    k = 0.425
    t = 1.27 * 2 #thickness

    sigma = calculate_sigma(k, t, x[0])
    sigma_crit.append(sigma)
        
    sigma = calculate_sigma(k, t, Length - x[-1])
    sigma_crit.append(sigma)
    
    print("Sigma Crit for Case2: ")
    for sig in sigma_crit:
        print(round(sig, 4), end = " ")
    print()
    if not(all(sig >= 6 for sig in sigma_crit)):
        print("Case 2 Fails. The bridge is under thin plate buckling")  
    print()
    sigma_crit.clear()

    #Case 3 k =6
    k = 6
    t = 1.27
        
    for b in y_top:
        sigma = calculate_sigma(k, t, b)
        sigma_crit.append(sigma)
    
    print("Sigma Crit for Case3: ")
    for sig in sigma_crit:
        print(round(sig, 4), end = " ")
    print()
    if not(all(sig >= 6 for sig in sigma_crit)):
        print("Case 3 Fails. The bridge is under thin plate buckling")  
    print()
    sigma_crit.clear()    
    #Case 4 
    k = 5
    t = 1.27
    
    for a, h in zip(a, height):
        sigma = calculate_sigma(k, t, a) + calculate_sigma(k, t, h)
        sigma_crit.append(sigma)
        

    print("Sigma Crit for Case 4: ")
    for sig in sigma_crit:
        print(sig, end = " ")
    print()
    if not(all(sig >= 6 for sig in sigma_crit)):
        print("Case 4 Fails. The bridge is under thin plate buckling")  
    sigma_crit.clear()
    


def section_properties_left():
    
    #x = np.array([0, 15, 16, 549, 550, 787])  # Location, x, of cross-section change
    #tfb = np.array([100, 100, 100, 100, 100, 100])  # Top Flange Width
    #tft = np.array([2.54, 2.54, 2.54, 2.54, 2.54, 2.54])  # Top Flange Thickness
    #wh = np.array([110, 110, 110, 110, 110, 110])  # Web Height
    #wt = np.array([1.27, 1.27, 1.27, 1.27, 1.27, 1.27])  # Web Thickness (Assuming 2 separate webs)
    #ws = np.array([70, 70, 70, 70, 70, 70])  # Web Spacing
    #bfb = np.array([0, 0, 0, 0, 0, 0])  # Bottom Flange Width
    #bft = np.array([0, 0, 0, 0, 0, 0])  # Bottom Flange Thickness
    #gtb = np.array([10, 10, 10, 10, 10, 10])  # Glue Tab Width
    #gtt = np.array([1.27, 1.27, 1.27, 1.27, 1.27, 1.27])  # Glue Tab Thickness
    #a = np.array([30, 30, 260, 260, 160, 160])  # Diaphragm Spacing
    
    tfb = np.array([120])
    tfb = np.repeat(tfb, 4)
    
    tft = np.array([1.27 * 2])
    tft = np.repeat(tft, 4)
    
    wh = np.array([100])
    wh = np.repeat(wh, 4)
    
    wt = np.array([1.27])
    wt = np.repeat(wt, 4)
    
    bfb = np.array([0])
    bfb = np.repeat(bfb, 4)
    
    bft = np.array([0])
    bft = np.repeat(bft, 4)
    
    gtb = np.array([10])
    gtb = np.repeat(gtb, 4)
    
    gtt = np.array([1.27])
    gtt = np.repeat(gtt, 4)
        
    b = wt * 2
    n = len(tfb)
    ovheight = tft + wh + bft  # Total height along beam

    # Initialize arrays for area (A)
    tfa = np.zeros(n)
    bfa = np.zeros(n)
    wa = np.zeros(n)
    gta = np.zeros(n)

    # Initialize arrays for I_0
    tfI = np.zeros(n)
    bfI = np.zeros(n)
    wI = np.zeros(n)
    gtI = np.zeros(n)

    # Initialize arrays for sum of I_0
    tfIsum = np.zeros(n)
    bfIsum = np.zeros(n)
    wIsum = np.zeros(n)
    gtIsum = np.zeros(n)

    # Initialize ybar & I
    ybot = np.zeros(n)
    ytop = np.zeros(n)
    I = np.zeros(n)

    # Initialize Q sum values, Qcent, Qglue
    wQ = np.zeros(n)
    Qcent = np.zeros(n)
    Qglue = np.zeros(n)

    for i in range(n):
        # Areas, A
        tfa[i] = tfb[i] * tft[i]
        bfa[i] = bfb[i] * bft[i]
        wa[i] = 2 * wh[i] * wt[i]  # Assumes 2 webs
        gta[i] = 2 * gtb[i] * gtt[i]  # Assumes 2 glue tabs

        # Local centroids, y
        tfy = ovheight - (tft / 2)
        bfy = bft / 2
        wy = (wh / 2) + bft
        gty = ovheight[i] - tft[i] - (gtt / 2)

        # Global centroid, ybar
        ybot[i] = (tfa[i] * tfy[i] + bfa[i] * bfy[i] + wa[i] * wy[i] +
                   gta[i] * gty[i]) / (tfa[i] + bfa[i] + wa[i] + gta[i])
        ytop[i] = abs(ybot[i] - ovheight[i])

        # I naught, I_0
        tfI[i] = (tfb[i] * (tft[i]**3)) / 12
        bfI[i] = (bfb[i] * (bft[i]**3)) / 12
        wI[i] = 2 * (wt[i] * (wh[i]**3)) / 12
        gtI[i] = 2 * (gtb[i] * (gtt[i]**3)) / 12

        tfIsum[i] = tfI[i] + tfa[i] * (abs(ybot[i] - tfy[i]))**2
        bfIsum[i] = bfI[i] + bfa[i] * (abs(ybot[i] - bfy[i]))**2
        wIsum[i] = wI[i] + wa[i] * (abs(ybot[i] - wy[i]))**2
        gtIsum[i] = gtI[i] + gta[i] * (abs(ybot[i] - gty[i]))**2

        # Second moment of area, I
        I[i] = tfIsum[i] + bfIsum[i] + wIsum[i] + gtIsum[i]

        # First moment of area, Q (Qcent & Qglue)
        wQ[i] = (2 * wt[i] * (ybot[i] - bft[i])) * ((ybot[i] - bft[i]) / 2)
        Qcent[i] = bfa[i] * (ybot[i] - (bft[i] / 2)) + wQ[i]
        Qglue[i] = tfa[i] * (ovheight[i] - ybot[i] - (tft[i] / 2))

    return ybot, ytop, I, Qcent, Qglue, b, ovheight  #y_Top is compression y_bottom tension, 




if __name__ == '__main__':
    print("-----------------------------------")
    geography()
    print("-----------------------------------")
