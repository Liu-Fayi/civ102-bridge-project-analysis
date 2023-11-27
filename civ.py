import matplotlib.pyplot as plt
import math
import numpy as np

E = 4000
Length = 220
mu = 0.2  # Ratio


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

    SFDs = []
    BMDs = []
    for i in range(len(start_x)):
        SFD = [A_y[i]]
        BMD = [0]
        for j in range(1, 1201):
            SFD.append(SFD[j - 1])
            if (j - start_x[i]) in x_train:
                indx = x_train.index(j - start_x[i])
                SFD[j] = SFD[j] - p_train[indx]
                BMD.append(BMD[j - 1])
            else:
                BMD.append(BMD[j - 1] + SFD[j])

        plt.xlim(0, 1201)
        graph, (plt1, plt2) = plt.subplots(1, 2)
        points = [start_x[i] + x for x in x_train]

        BMD = [-1 * ele for ele in BMD]

        x = range(0, 1201)

        plt1.plot(x, SFD)
        plt1.plot(points, [SFD[x.index(p)] for p in points], 'ro', label='Highlighted Points')  # 'ro' means red circles
        plt1.set_title("SFD")

        plt2.plot(x, BMD)
        plt2.plot(points, [BMD[x.index(p)] for p in points], 'ro', label='Highlighted Points')  # 'ro' means red circles
        plt2.set_title("BMD")

        graph.tight_layout()
        plt.show()


def geography():
    # Setting up shear force,
    # I used max force for this one
    V = 300.2723333333333  # This value is obtained when  length is 1260mm
    M = 77199.71466666664

    # Location of each cross section:
    y_bot, y_top, I, Qcent, Qglue, b, height = section_properties_left()

    # Stress part calculation
    bridge_shear = V * Qcent / I / b
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
    print(
        f"Max Bridge Shear: {round(max_bridge_shear, 4)} Max Glue Shear: {round(max_glue_shear, 4)} \nMax Compression: {round(max_compression, 4)} Max tension: {round(max_tension, 4)}")
    print("-----------------------------------")
    plate_buckling(y_top, height)


# Take k t(thickness) and b to calculat the sigma_crit
def calculate_sigma(k, t, b):
    return k * math.pi ** 2 * E / (12 * (1 - mu ** 2)) * (t / b) ** 2


def plate_buckling(y_top, height):
    sigma_crit = []
    # Location of cross section
    x = [20, 100, 180, 280]
    # Location of diagphram
    a = [130, 300, 300, 130]

    # Case 1  k = 4
    k = 4
    t = 1.27 * 2  # thickness

    for i in range(1, len(x)):
        b = x[i] - x[i - 1]
        sigma = calculate_sigma(k, t, b)
        sigma_crit.append(sigma)

    print("Sigma Crit for Case1: ")
    for sig in sigma_crit:
        print(round(sig, 4), end=" ")
    print()
    if not (all(sig >= 6 for sig in sigma_crit)):
        print("Case 1 Fails. The bridge is under thin plate buckling")
    print()
    # Case 2 k = 0.425
    k = 0.425
    t = 1.27 * 2  # thickness

    sigma = calculate_sigma(k, t, x[0])
    sigma_crit.append(sigma)

    sigma = calculate_sigma(k, t, Length - x[-1])
    sigma_crit.append(sigma)

    print("Sigma Crit for Case2: ")
    for sig in sigma_crit:
        print(round(sig, 4), end=" ")
    print()
    if not (all(sig >= 6 for sig in sigma_crit)):
        print("Case 2 Fails. The bridge is under thin plate buckling")
    print()
    sigma_crit.clear()

    # Case 3 k =6
    k = 6
    t = 1.27

    for b in y_top:
        sigma = calculate_sigma(k, t, b)
        sigma_crit.append(sigma)

    print("Sigma Crit for Case3: ")
    for sig in sigma_crit:
        print(round(sig, 4), end=" ")
    print()
    if not (all(sig >= 6 for sig in sigma_crit)):
        print("Case 3 Fails. The bridge is under thin plate buckling")
    print()
    sigma_crit.clear()
    # Case 4
    k = 5
    t = 1.27

    for a, h in zip(a, height):
        sigma = calculate_sigma(k, t, a) + calculate_sigma(k, t, h)
        sigma_crit.append(sigma)

    print("Sigma Crit for Case 4: ")
    for sig in sigma_crit:
        print(sig, end=" ")
    print()
    if not (all(sig >= 6 for sig in sigma_crit)):
        print("Case 4 Fails. The bridge is under thin plate buckling")
    sigma_crit.clear()


def section_properties_left():
    # x = np.array([0, 15, 16, 549, 550, 787])  # Location, x, of cross-section change
    # tfb = np.array([100, 100, 100, 100, 100, 100])  # Top Flange Width
    # tft = np.array([2.54, 2.54, 2.54, 2.54, 2.54, 2.54])  # Top Flange Thickness
    # wh = np.array([110, 110, 110, 110, 110, 110])  # Web Height
    # wt = np.array([1.27, 1.27, 1.27, 1.27, 1.27, 1.27])  # Web Thickness (Assuming 2 separate webs)
    # ws = np.array([70, 70, 70, 70, 70, 70])  # Web Spacing
    # bfb = np.array([0, 0, 0, 0, 0, 0])  # Bottom Flange Width
    # bft = np.array([0, 0, 0, 0, 0, 0])  # Bottom Flange Thickness
    # gtb = np.array([10, 10, 10, 10, 10, 10])  # Glue Tab Width
    # gtt = np.array([1.27, 1.27, 1.27, 1.27, 1.27, 1.27])  # Glue Tab Thickness
    # a = np.array([30, 30, 260, 260, 160, 160])  # Diaphragm Spacing

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
        tfI[i] = (tfb[i] * (tft[i] ** 3)) / 12
        bfI[i] = (bfb[i] * (bft[i] ** 3)) / 12
        wI[i] = 2 * (wt[i] * (wh[i] ** 3)) / 12
        gtI[i] = 2 * (gtb[i] * (gtt[i] ** 3)) / 12

        tfIsum[i] = tfI[i] + tfa[i] * (abs(ybot[i] - tfy[i])) ** 2
        bfIsum[i] = bfI[i] + bfa[i] * (abs(ybot[i] - bfy[i])) ** 2
        wIsum[i] = wI[i] + wa[i] * (abs(ybot[i] - wy[i])) ** 2
        gtIsum[i] = gtI[i] + gta[i] * (abs(ybot[i] - gty[i])) ** 2

        # Second moment of area, I
        I[i] = tfIsum[i] + bfIsum[i] + wIsum[i] + gtIsum[i]

        # First moment of area, Q (Qcent & Qglue)
        wQ[i] = (2 * wt[i] * (ybot[i] - bft[i])) * ((ybot[i] - bft[i]) / 2)
        Qcent[i] = bfa[i] * (ybot[i] - (bft[i] / 2)) + wQ[i]
        Qglue[i] = tfa[i] * (ovheight[i] - ybot[i] - (tft[i] / 2))

    return ybot, ytop, I, Qcent, Qglue, b, ovheight  # y_Top is compression y_bottom tension,


def single_cross_section_calc(L, G_heights_top):
    """input: L is a list of lists of all rectangular cross-section members (which varies over the
            length of the bridge) of the bridge
            the inner lists describe each member as [x, y, width, height]
            x and y are the coordinates of the top left corner of the rectangle (all in mm)"""

    y_bar = 0  # y_bar is the centroid height of the cross-section
    I = 0  # I is the second moment of area of the cross-section
    y_top = 0  # y_top is the distance from the centroid to the top of the cross-section
    Q_cent = 0  # Q_cent is the first moment of area of the cross-section at centroid height
    Q_glue = [0] * len(G_heights_top)  # Q_glue is the first moment of area of the cross-section at glue height
    b_glue = [0] * len(G_heights_top)  # b_glue is the width of the cross-section at glue height
    b_glue_top = [0] * len(G_heights_top)
    b_glue_bot = [0] * len(G_heights_top)
    b_cent = 0  # b_cent is the width of the cross-section at centroid height
    ovheight = 0  # ovheight is the overall height of the cross-section
    for member in L:
        if member[1] + member[3] > ovheight:
            ovheight = member[1] + member[3]  # calculating the overall height of the cross-section
    A = 0  # A is the total cross-sectional area (used to calculate y_bar)
    hA = 0  # hA is the sum of the areas times the distance from the centroid of each area to the bottom of the
    # cross-section (used to calculate y_bar)
    for member in L:  # centroid height calculations
        A += member[2] * member[3]
        hA += member[2] * member[3] * (ovheight - (member[1] + member[3] / 2))
    y_bar = hA / A
    y_top = ovheight - y_bar

    for member in L:
        I += member[2] * member[3] ** 3 / 12 + member[2] * member[3] * (
                ovheight - (member[1] + member[3] / 2) - y_bar) ** 2  # second moment of area calculations (parallel
        # axis theorem)

        if member[1] + member[3] < y_top:  # first moment of area calculations
            Q_cent += member[2] * member[3] * (y_top - member[1] + member[3] / 2)
        elif member[1] < y_top:
            Q_cent += member[2] * (y_top - member[1]) * ((y_top - member[1]) / 2)

        if member[1] + member[3] > y_top > member[1]:
            b_cent += member[2]

        for i in range(len(G_heights_top)):
            if member[1] + member[3] <= G_heights_top[i]:
                Q_glue[i] += member[2] * member[3] * (y_top - (member[1] + member[3] / 2))
            if member[1] == G_heights_top[i]:
                b_glue_bot[i] += member[2]
            elif member[1] + member[3] == G_heights_top[i]:
                b_glue_top[i] += member[2]

    for i in range(len(G_heights_top)):
        b_glue[i] = min(b_glue_top[i], b_glue_bot[i])

    return y_bar, y_top, I, Q_cent, Q_glue, b_cent, ovheight, b_glue


def total_cross_section_calc(L, G_heights_top):
    """input: L is a list of lists of all cross_section list of rectangular cross-section members
            (which varies over the length of the bridge) of the bridge
            the inner lists describe each member as [x, y, width, height]
            x and y are the coordinates of the top left corner of the rectangle (all in mm)"""
    I = [0] * len(L)
    y_bar = [0] * len(L)
    y_top = [0] * len(L)
    Q_cent = [0] * len(L)
    Q_glue = [None] * len(L)
    b_cent = [0] * len(L)
    ovheight = [0] * len(L)
    b_glue = [None] * len(L)
    for i in range(len(L)):  # calculating cross-section properties for each cross-section using
        # single_cross_section_calc
        y_bar[i], y_top[i], I[i], Q_cent[i], Q_glue[i], b_cent[i], ovheight[i], b_glue[i] = single_cross_section_calc(
            L[i], G_heights_top)
    return y_bar, y_top, I, Q_cent, Q_glue, b_cent, ovheight, b_glue


def stress_demand_calc(L, s_t=30, s_c=6, t=4):
    """input: L is a list of lists of all cross_section properties as outputed by total_cross_section_calc"""
    M_t_allow = [0] * len(L[0])  # M_t_allow is the allowable moment for tension side of each cross-section
    M_c_allow = [0] * len(L[0])  # M_c_allow is the allowable moment for compression side of each cross-section
    V_allow = [0] * len(L[0])  # V_allow is the allowable shear force for each cross-section
    V_glue_allow = []  # V_glue_allow is a list of allowable shear force for each glue tab
    for i in range(len(L[0])):
        M_t_allow[i] = s_t * L[2][i] / L[0][i]  # calculating allowable moment for tension side of each cross-section
        # (Navier's equation)
        M_c_allow[i] = s_c * L[2][i] / L[1][i]  # calculating allowable moment for compression side of each
        # cross-section (Navier's equation)
        V_allow[i] = t * L[2][i] * L[5][i] / L[3][i]  # calculating allowable shear force for each cross-section
        # (tau = VQ/Ib)
        for j in range(len(L[4][i])):
            V_glue_allow.append(2 * L[2][i] * L[7][i][j] / L[4][i][j])  # calculating allowable shear force for each
            # glue tab (tau = VQ/Ib)

    return M_t_allow, M_c_allow, V_allow, V_glue_allow


def stress_buckling_crit_calc(open_flange_width, closed_flange_width, y_top, web_height, diaphragm_spacing,
                              horizontal_height=1.27, vertical_thickness=1.27):
    """return the critical buckling stresses for each thin plate buckling cases"""
    closed_flange_crit = 4 * math.pi ** 2 * E / (12 * (1 - mu ** 2)) * (horizontal_height / closed_flange_width) ** 2
    # case 1
    open_flange_crit = 0.425 * math.pi ** 2 * E / (12 * (1 - mu ** 2)) * (horizontal_height / open_flange_width) ** 2
    # case 2
    web_crit = 6 * math.pi ** 2 * E / (12 * (1 - mu ** 2)) * (vertical_thickness / y_top) ** 2  # case 3
    shear_crit = 5 * math.pi ** 2 * E / (12 * (1 - mu ** 2)) * (
            ((vertical_thickness / web_height) ** 2) + ((vertical_thickness / diaphragm_spacing) ** 2))  # case 4
    return closed_flange_crit, open_flange_crit, web_crit, shear_crit


def design0():
    dim = total_cross_section_calc([[[0, 0, 100, 1.27], [10, 1.27, 1.27, 75], [90 - 1.27, 1.27, 1.27, 75],
                                     [10 + 1.27, 75, 80 - 2 * 1.27, 1.27], [10 + 1.27, 1.27, 5, 1.27],
                                     [85 - 1.27, 1.27, 5, 1.27]]], [1.27])
    print(dim)
    print("y_top ratio")
    print(dim[1][0] / dim[6][0])
    print("calculating critical buckling stresses")
    buckling_crit = stress_buckling_crit_calc(10, 70 - 1.27 * 2, dim[1][0], 75, 400)
    print(buckling_crit)
    stress_demand_max = stress_demand_calc(dim, 30,
                                           min(6, min(buckling_crit[0], min(buckling_crit[1], buckling_crit[2]))),
                                           min(4, buckling_crit[3]))
    print(stress_demand_max)
    print(stress_demand_max[0][0] / 568, stress_demand_max[1][0] / 568, stress_demand_max[2][0] / 1.774,
          stress_demand_max[3][0] / 1.774)
    print(3.35 * min(stress_demand_max[0][0] / 568, min(stress_demand_max[1][0] / 568,
                                                        min(stress_demand_max[2][0] / 1.774,
                                                            stress_demand_max[3][0] / 1.774))))
    print("FOS")
    print("axial tension:", stress_demand_max[0][0] / 69792.5)
    print("axial compression:", stress_demand_max[1][0] / 69792.5)
    print("shear:", min(stress_demand_max[2][0] / 258.285, stress_demand_max[3][0] / 258.285))


def design1():
    L = [[[0, 0, 100, 1.27], [25, 1.27, 1.27, 200], [75 - 1.27, 1.27, 1.27, 200],
          [50 - 1.27 / 2, 1.27, 1.27, 150]]]
    print("defining shapes")
    print("calculating geometric properties")
    dim = total_cross_section_calc(L, [1.27])
    print(dim)
    print("y top ratio")
    print(dim[1][0] / dim[6][0])
    print("calculating critical buckling stresses")
    buckling_crit = stress_buckling_crit_calc(10, 80 - 1.27 * 2, max(dim[1]), 500, 300)
    print(buckling_crit)
    print("calculating stress demands")
    stress_demand_max = stress_demand_calc(dim)
    print(stress_demand_max)
    # num = 1
    # # 1: max tensile moment
    # # 2: max compressive moment
    # # 3: max shear force
    # # 4: max glue shear force
    # for i in stress_demand_max:
    #     print(num)
    #     print(i[0])
    #     plt.plot(range(1250), i)
    #     plt.ylabel(num)
    #     plt.show()
    #     num += 1
    real_max = stress_demand_calc(dim, 30, min(6, min(buckling_crit[0], min(buckling_crit[1], buckling_crit[2]))),
                                  min(4, buckling_crit[3]))
    print(real_max)
    print(real_max[0][0] / 568, real_max[1][0] / 568, real_max[2][0] / 1.774, real_max[3][0] / 1.774)
    print(3.35 * min(real_max[0][0] / 568,
                     min(real_max[1][0] / 568, min(real_max[2][0] / 1.774, real_max[3][0] / 1.774))))


def design2():
    L = [None] * 1250
    print("defining shapes")
    for l in range(25):
        L[l] = [[0, 0, 100, 1.27], [10, 1.27, 1.27, 200], [90 - 1.27, 1.27, 1.27, 200]]
    for l in range(25, 100):
        L[l] = [[0, 0, 100, 1.27], [10, 1.27, 1.27, -0.2917 * l + 207.2917],
                [90 - 1.27, 1.27, 1.27, -0.2917 * l + 207.2917]]
    for l in range(100, 625):
        L[l] = [[0, 0, 100, 1.27], [10, 1.27, 1.27, -0.2917 * l + 207.2917],
                [90 - 1.27, 1.27, 1.27, -0.2917 * l + 207.2917], [50 - 1.27 / 2, 1.27, 1.27, 0.9524 * l - 95.2381]]
    for l in range(625, 1150):
        L[l] = [[0, 0, 100, 1.27], [10, 1.27, 1.27, 0.2917 * l - 157.2917],
                [90 - 1.27, 1.27, 1.27, 0.2917 * l - 157.2917], [50 - 1.27 / 2, 1.27, 1.27, -0.9524 * l + 1095.24]]
    for l in range(1150, 1225):
        L[l] = [[0, 0, 100, 1.27], [10, 1.27, 1.27, 0.2917 * l - 157.2917],
                [90 - 1.27, 1.27, 1.27, 0.2917 * l - 157.2917]]
    for l in range(1225, 1250):
        L[l] = [[0, 0, 100, 1.27], [10, 1.27, 1.27, 200], [90 - 1.27, 1.27, 1.27, 200]]
    print("calculating geometric properties")
    dim = total_cross_section_calc(L, [1.27])
    print("y top ratio")
    for i in range(len(dim[1])):
        print(dim[1][i] / dim[6][i])
    plt.plot(range(1250), dim[1])
    print("calculating critical buckling stresses")
    buckling_crit = stress_buckling_crit_calc([10, 80 - 1.27 * 2, max(dim[1]), 500, 300])
    print(buckling_crit)
    print("calculating stress demands")
    stress_demand_max = stress_demand_calc(dim)
    num = 1
    # 1: max tensile moment
    # 2: max compressive moment
    # 3: max shear force
    # 4: max glue shear force
    for i in stress_demand_max:
        plt.plot(range(1250), i)
        plt.ylabel(num)
        plt.show()
        num += 1

    real_max = stress_demand_calc(dim, 30, min(6, min(buckling_crit[0], min(buckling_crit[1], buckling_crit[2]))),
                                  min(4, buckling_crit[3]))
    num = 1
    for i in real_max:
        plt.plot(range(1250), i)
        plt.ylabel(num)
        plt.show()
        num += 1


def design3():
    dim = total_cross_section_calc([[[0, 0, 100, 1.27 * 3], [15 - 1.27, 1.27 * 3, 1.27, 105], [85, 1.27 * 3, 1.27, 105],
                                     [15, 1.27 * 3, 10, 1.27], [75 - 1.27, 1.27 * 3, 10, 1.27]]], [1.27 * 3])
    print(dim)
    print("y_top ratio")
    print(dim[1][0] / dim[6][0])
    print("calculating critical buckling stresses")
    buckling_crit = stress_buckling_crit_calc(15 - 1.27, 70, dim[1][0], 105, 140, horizontal_height=1.27 * 3)
    print(buckling_crit)
    stress_demand_max = stress_demand_calc(dim)
    print(stress_demand_max)
    real_max = stress_demand_calc(dim, 30, min(6, min(buckling_crit[0], min(buckling_crit[1], buckling_crit[2]))),
                                  min(4, buckling_crit[3]))
    print(real_max)
    print(real_max[0][0] / 568, real_max[1][0] / 568, real_max[2][0] / 1.774, real_max[3][0] / 1.774)
    print(3.35 * min(real_max[0][0] / 568,
                     min(real_max[1][0] / 568, min(real_max[2][0] / 1.774, real_max[3][0] / 1.774))))


def design4():
    geo = [[0, 0, 100, 1.27], [0, 1.27, 100, 1.27], [40, 5 + 1.27, 20, 1.27], [10, 1.27 * 2, 10, 1.27],
           [80, 1.27 * 2, 10, 1.27]]
    for i in range(100):
        geo.append([20 + i / 5, 1.27 * 2 + i / 100 * (5 - 1.27), 1 / 5, 1.29162843758])
        geo.append([80 - (i + 1) / 5, 1.27 * 2 + i / 100 * (5 - 1.27), 1 / 5, 1.29162843758])
    dim = total_cross_section_calc([geo], [1.27])
    print(dim)
    print("y_top ratio")
    print(dim[1][0] / dim[6][0])
    # print("calculating critical buckling stresses")
    # buckling_crit = stress_buckling_crit_calc(20, 80 - 1.27, dim[1], 1.27*(3**0.5), 200)
    stress_demand_max = stress_demand_calc(dim)
    print(stress_demand_max)
    print(stress_demand_max[0][0] / 568, stress_demand_max[1][0] / 568, stress_demand_max[2][0] / 1.774, stress_demand_max[3][0] / 1.774)
    print(3.35 * min(stress_demand_max[0][0] / 568, min(stress_demand_max[1][0] / 568, min(stress_demand_max[2][0] / 1.774, stress_demand_max[3][0] / 1.774))))


def design5():
    dim = total_cross_section_calc([[[0, 0, 166, 1.27], [0, 1.27, 166, 1.27],
                                     [37.5 - 1.27, 1.27 * 2, 1.27, 100 - 1.27 * 2],
                                     [112.5, 1.27 * 2, 1.27, 100 - 1.27 * 2],
                                     [30, 1.27 * 2, 15, 1.27], [105, 1.27 * 2, 15, 1.27]]], [1.27 * 2, 1.27])
    dim[7][0][1] = 2
    print(dim)
    print("y_top ratio")
    print(dim[1][0] / dim[6][0])
    print("calculating critical buckling stresses")
    buckling_crit = stress_buckling_crit_calc(37.5 - 1.27, 75, dim[1][0], 100 - 1.27 * 2, 156.25,
                                              horizontal_height=1.27 * 2)
    print(buckling_crit)
    stress_demand_max = stress_demand_calc(dim)
    print(stress_demand_max)
    real_max = stress_demand_calc(dim, 30, min(6, min(buckling_crit[0], min(buckling_crit[1], buckling_crit[2]))),
                                  min(4, buckling_crit[3]))
    print(real_max)
    print(real_max[0][0] / 568, real_max[1][0] / 568, real_max[2][0] / 1.774, real_max[3][0] / 1.774,
          real_max[3][1] / 1.774)
    print(3.35 * min(real_max[0][0] / 568, min(real_max[1][0] / 568, min(real_max[2][0] / 1.774,
                                                                         min(real_max[3][0] / 1.774,
                                                                             real_max[3][1] / 1.774)))))


def design5_splice():
    dim = total_cross_section_calc([[[0, 0, 166, 1.27], [0, 1.27, 166, 1.27],
                                     [30 - 1.27, 1.27 * 2, 1.27, 100 - 1.27 * 2], [120, 1.27 * 2, 1.27, 100 - 1.27 * 2],
                                     [30, 1.27 * 2, 15, 1.27], [105, 1.27 * 2, 15, 1.27], [120 + 1.27, 100 - 52, 1.27,
                                                                                           52]]], [1.27 * 2, 1.27])
    dim[7][0][1] = 2
    print(dim)
    print("y_top ratio")
    print(dim[1][0] / dim[6][0])
    print("calculating critical buckling stresses")
    buckling_crit = stress_buckling_crit_calc(37.5 - 1.27, 75, dim[1][0], 100 - 1.27 * 2, 156.25,
                                              horizontal_height=1.27 * 2)
    print(buckling_crit)
    stress_demand_max = stress_demand_calc(dim)
    print(stress_demand_max)
    real_max = stress_demand_calc(dim, 30, min(6, min(buckling_crit[0], min(buckling_crit[1], buckling_crit[2]))),
                                  min(4, buckling_crit[3]))
    print(real_max)
    print(real_max[0][0] / 568, real_max[1][0] / 568, real_max[2][0] / 1.774, real_max[3][0] / 1.774,
          real_max[3][1] / 1.774)
    print(3.35 * min(real_max[0][0] / 568, min(real_max[1][0] / 568, min(real_max[2][0] / 1.774,
                                                                         min(real_max[3][0] / 1.774,
                                                                             real_max[3][1] / 1.774)))))


def design5p():
    dim = total_cross_section_calc([[[-25, 0, 175, 1.27], [0, 1.27, 143, 1.27],
                                     [30 - 1.27, 1.27 * 2, 1.27, 100 - 1.27 * 2], [120, 1.27 * 2, 1.27, 100 - 1.27 * 2],
                                     [30, 1.27 * 2, 15, 1.27], [105, 1.27 * 2, 15, 1.27]]], [1.27 * 2, 1.27])
    print(dim)
    print("y_top ratio")
    print(dim[1][0] / dim[6][0])
    print("calculating critical buckling stresses")
    buckling_crit = stress_buckling_crit_calc(37.5 - 1.27, 75, dim[1][0], 100 - 1.27 * 2, 150,
                                              horizontal_height=1.27 * 2)
    print(buckling_crit)
    stress_demand_max = stress_demand_calc(dim)
    print(stress_demand_max)
    real_max = stress_demand_calc(dim, 30, min(6, min(buckling_crit[0], min(buckling_crit[1], buckling_crit[2]))),
                                  min(4, buckling_crit[3]))
    print(real_max)
    print(real_max[0][0] / 568, real_max[1][0] / 568, real_max[2][0] / 1.774, real_max[3][0] / 1.774)
    print(3.35 * min(real_max[0][0] / 568,
                     min(real_max[1][0] / 568, min(real_max[2][0] / 1.774, real_max[3][0] / 1.774))))


if __name__ == '__main__':
    # design5()
    # print("-----------------------------------")
    # design5_splice()
    design0()
    print("-----------------------------------")
    design1()
    print("-----------------------------------")
    design5()
    print("-----------------------------------")
    design4()
