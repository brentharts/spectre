#!/usr/bin/python3
import numpy as np
from time import time

## configlation
#* increase this number for larger tilings.
N_ITERATIONS = 4
#* shape Edge_ration tile(Edge_a, Edge_b)
Edge_a = 10.0 # 20.0 / (np.sqrt(3) + 1.0)
Edge_b = 10.0 # 20.0 - Edge_a
## end of configilation.

TILE_NAMES = ["Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi"]

def get_spectre_points(edge_a, edge_b):
    a = edge_a
    a_sqrt3_d2 = a * np.sqrt(3)/2 # a*sin(60 deg)
    a_d2 = a * 0.5  # a* cos(60 deg)

    b = edge_b
    b_sqrt3_d2 = b * np.sqrt(3) / 2 # b*sin(60 deg)
    b_d2 = b * 0.5 # b* cos(60 deg)

    spectre_points = np.array([
		(0                        , 0                            ), #// 1: - b
		(a                        , 0                            ), #// 2: + a
		(a +     a_d2             , 0 - a_sqrt3_d2               ), #// 3: + ~a
		(a +     a_d2 + b_sqrt3_d2, 0 - a_sqrt3_d2 +         b_d2), #// 4: + ~b
		(a +     a_d2 + b_sqrt3_d2, 0 - a_sqrt3_d2 +     b + b_d2), #// 5: + b
		(a + a + a_d2 + b_sqrt3_d2, 0 - a_sqrt3_d2 +     b + b_d2), #// 6: + a
		(a + a + a +    b_sqrt3_d2,                      b + b_d2), #// 7: + ~a
		(a + a + a                ,                  b + b       ), #// 8: - ~b 
		(a + a + a    - b_sqrt3_d2,                  b + b - b_d2), #// 9: - ~b
		(a + a + a_d2 - b_sqrt3_d2,     a_sqrt3_d2 + b + b - b_d2), #// 10: +~a
		(a +     a_d2 - b_sqrt3_d2,     a_sqrt3_d2 + b + b - b_d2), #// 11: -a
		(        a_d2 - b_sqrt3_d2,     a_sqrt3_d2 + b + b - b_d2), #// 12: -a
		(0            - b_sqrt3_d2,                  b + b - b_d2), #// 13: -~a
		(0                        ,                      b       )  #// 14: +~b
    ], 'float32')
    # print(spectre_points)
    return spectre_points
   
SPECTRE_POINTS = get_spectre_points(Edge_a, Edge_b) # tile(Edge_a, Edge_b)
Mystic_SPECTRE_POINTS = get_spectre_points(Edge_b, Edge_a) # tile(Edge_b, Edge_a)
SPECTRE_QUAD = SPECTRE_POINTS[[3,5,7,11],:]

PI = np.pi
IDENTITY = np.array([[1,0,0],[0,1,0]], 'float32') # == trot(0)

# Rotation matrix
trot_memo = {
     0:  np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0]]),
     30: np.array([[np.sqrt(3)/2, -0.5, 0.0], [0.5, np.sqrt(3)/2, 0.0]]),
     60: np.array([[0.5, -np.sqrt(3)/2, 0.0], [np.sqrt(3)/2, 0.5, 0.0]]),
     120: np.array([[-0.5, -np.sqrt(3)/2, 0.0], [np.sqrt(3)/2, -0.5, 0.0]]),
     180: np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
     240:  np.array([[-0.5, np.sqrt(3)/2, 0.0], [-np.sqrt(3)/2, -0.5, 0.0]]),
     -120: np.array([[-0.5, np.sqrt(3)/2, 0.0], [-np.sqrt(3)/2, -0.5, 0.0]]),
}
def trot(degAngle):
    """
    degAngle: integer degree angle 
    """
    global trot_memo
    if degAngle not in trot_memo:
        ang = np.deg2rad(degAngle)
        c = np.cos(ang)
        s = np.sin(ang)
        trot_memo[degAngle] = np.array([[c, -s, 0],[s, c, 0]])
        print(f"trot_memo[{degAngle}]={trot_memo[degAngle]}")
    return trot_memo[degAngle].copy()

trot_replace_arr = np.array([-1, -0.8660254037844386, -0.5, 0, 0.5, 0.8660254037844386, 1])
def trot_refine(trsf):
    """
    trsf: transformation matrix
    """
    idx = np.abs(np.subtract.outer(trsf[:2, :2].flatten(), trot_replace_arr)).argmin(1)
    trsf[:2, :2] = trot_replace_arr[idx].reshape(trsf[:2, :2].shape)
    return trsf

# Matrix * point
def transPt(trsf, quad):
    return  (trsf[:,:2].dot(quad) + trsf[:,2])

# Matrix * point
def mul(A, B):
    AB = A.copy()
    AB[:,:2] = A[:,:2].dot(B[:,:2]) 
    AB[:,2] += A[:,:2].dot(B[:,2])
    return AB

class Tile:
    def __init__(self, label):
        """
        _: NO list of Tile coordinate points
        label: Tile type used for shapes coloring
        """
        self.label = label
        self.quad = SPECTRE_QUAD

    def drawPolygon(self, drawer, tile_transformation=IDENTITY):
        return drawer(tile_transformation, self.label)

class MetaTile:
    def __init__(self, tiles=[], transformations=[], quad=SPECTRE_QUAD):
        """
        tiles: list of Tiles(No points)
        transformations: list of transformation matrices
        quad: MetaTile quad points
        """
        self.tiles = tiles
        self.transformations = transformations
        self.quad = quad

    def drawPolygon(self, drawer, transformation=IDENTITY):
        """
        recursively expand MetaTiles down to Tiles and draw those
        """
        # TODO: parallelize?
        for tile, trsf in zip(self.tiles, self.transformations):
           tile.drawPolygon(drawer,  trot_refine(mul(transformation, trsf)))
                            
def buildSpectreBase():
    tiles = {label: (Tile(label) ) for label in TILE_NAMES if label != "Gamma"}
    # special rule for Mystic == Gamma == Gamma1 + Gamma2
    tiles["Gamma"] = MetaTile(tiles=[Tile("Gamma1"),
                                     Tile("Gamma2")],
                                     transformations=[
                                         IDENTITY.copy(),
                                         mul(np.array([
                                             [1,0,SPECTRE_POINTS[8,0]],
                                             [0,1,SPECTRE_POINTS[8,1]]
                                         ]), trot(30))],
                                     quad=SPECTRE_QUAD.copy())
    return tiles

transformation_min = 0.0
transformation_max = 0.0
def buildSupertiles(input_tiles):
    """
    iteratively build on current system of tiles
    input_tiles = current system of tiles, initially built with buildSpectreBase()
    """
    # First, use any of the nine-unit tiles in "tiles" to obtain a
    # list of transformation matrices for placing tiles within supertiles.
    quad = input_tiles["Delta"].quad

    transformations = [IDENTITY.copy()]
    total_angle = 0
    rotation = IDENTITY.copy() # trot(total_angle)
    transformed_quad = quad
    for _angle, _from, _to in ((  60, 3, 1),
                               (   0, 2, 0),
                               (  60, 3, 1),
                               (  60, 3, 1),
                               (   0, 2, 0),
                               (  60, 3, 1),
                               (-120, 3, 3)):
        if _angle != 0:
            total_angle += _angle
            rotation = trot(total_angle)
            transformed_quad = quad.dot(rotation[:,:2].T) # + trot[:,2]
        ttrans = IDENTITY.copy()
        ttrans[:,2] = transPt(transformations[-1], quad[_from]) - transformed_quad[_to,:]
        transformations.append(mul(ttrans, rotation))

    R = np.array([[-1,0,0],[ 0,1,0]], 'float32')
    transformations = [ trot_refine(mul(R, trsf)) for trsf in transformations ]
    global transformation_min, transformation_max
    transformation_min = np.min(transformations)
    transformation_max = np.max(transformations)

    # Now build the actual supertiles, labeling appropriately.
    super_quad =  np.array([
        transPt(transformations[6], quad[2]),
        transPt(transformations[5], quad[1]),
        transPt(transformations[3], quad[2]),
        transPt(transformations[0], quad[1]) 
    ])

    tiles = {label: MetaTile(tiles=[input_tiles[subst] for subst in substitutions if subst],
                     transformations=[trsf for subst, trsf in zip(substitutions, transformations) if subst],
                     quad=super_quad
                     ) for label, substitutions in (
                         ("Gamma",  ("Pi",  "Delta", None,  "Theta", "Sigma", "Xi",  "Phi",    "Gamma")),
                         ("Delta",  ("Xi",  "Delta", "Xi",  "Phi",   "Sigma", "Pi",  "Phi",    "Gamma")),
                         ("Theta",  ("Psi", "Delta", "Pi",  "Phi",   "Sigma", "Pi",  "Phi",    "Gamma")),
                         ("Lambda", ("Psi", "Delta", "Xi",  "Phi",   "Sigma", "Pi",  "Phi",    "Gamma")),
                         ("Xi",     ("Psi", "Delta", "Pi",  "Phi",   "Sigma", "Psi", "Phi",    "Gamma")),
                         ("Pi",     ("Psi", "Delta", "Xi",  "Phi",   "Sigma", "Psi", "Phi",    "Gamma")),
                         ("Sigma",  ("Xi",  "Delta", "Xi",  "Phi",   "Sigma", "Pi",  "Lambda", "Gamma")),
                         ("Phi",    ("Psi", "Delta", "Psi", "Phi",   "Sigma", "Pi",  "Phi",    "Gamma")),
                         ("Psi",    ("Psi", "Delta", "Psi", "Phi",   "Sigma", "Psi", "Phi",    "Gamma"))
                      )}
    return tiles

#### main process ####
# global N_ITERATIONS ,Edge_a,Edge_b

start = time()
tiles = buildSpectreBase()
for _ in range(N_ITERATIONS):
    tiles = buildSupertiles(tiles)
transformation_min = int(np.floor(transformation_min))
transformation_max = int(np.ceil(transformation_max))
time1 = time()-start
print(f"transformation range is {transformation_min} to {transformation_max}")
print(f"supertiling loop took {round(time1, 4)} seconds")

### drawing parameter data
# Color map from Figure 5.3
COLOR_MAP = {
	'Gamma': np.array((203, 157, 126),'f')/255.,
	'Gamma1': np.array((203, 157, 126),'f')/255.,
	'Gamma2': np.array((203, 157, 126),'f')/255.,
	'Delta': np.array((163, 150, 133),'f')/255.,
	'Theta': np.array((208, 215, 150),'f')/255.,
	'Lambda': np.array((184, 205, 178),'f')/255.,
	'Xi': np.array((211, 177, 144),'f')/255.,
	'Pi': np.array((218, 197, 161),'f')/255.,
	'Sigma': np.array((191, 146, 126),'f')/255.,
	'Phi': np.array((228, 213, 167),'f')/255.,
	'Psi': np.array((224, 223, 156),'f')/255.
}

# COLOR_MAP_orig
COLOR_MAP = {
	'Gamma': np.array((255, 255, 255),'f')/255.,
	'Gamma1': np.array((255, 255, 255),'f')/255.,
	'Gamma2': np.array((255, 255, 255),'f')/255.,
	'Delta': np.array((220, 220, 220),'f')/255.,
	'Theta': np.array((255, 191, 191),'f')/255.,
	'Lambda': np.array((255, 160, 122),'f')/255.,
	'Xi': np.array((255, 242, 0),'f')/255.,
	'Pi': np.array((135, 206, 250),'f')/255.,
	'Sigma': np.array((245, 245, 220),'f')/255.,
	'Phi': np.array((0, 255, 0),'f')/255.,
	'Psi': np.array((0, 255, 255),'f')/255.
}

# COLOR_MAP_mystics 
COLOR_MAP = {
	'Gamma': np.array((196, 201, 169),'f')/255.,
	'Gamma1': np.array((196, 201, 169),'f')/255.,
	'Gamma2': np.array((156, 160, 116),'f')/255.,
	'Delta': np.array((247, 252, 248),'f')/255.,
	'Theta': np.array((247, 252, 248),'f')/255.,
	'Lambda': np.array((247, 252, 248),'f')/255.,
	'Xi': np.array((247, 252, 248),'f')/255.,
	'Pi': np.array((247, 252, 248),'f')/255.,
	'Sigma': np.array((247, 252, 248),'f')/255.,
	'Phi': np.array((247, 252, 248),'f')/255.,
	'Psi': np.array((247, 252, 248),'f')/255.
}

# COLOR_MAP_pride
COLOR_MAP = {
    "Gamma":  np.array((255, 255, 255),'f')/255.,
    "Gamma1": np.array(( 97,  57,  21),'f')/255.,
    "Gamma2": np.array(( 64,  64,  64),'f')/255.,
    "Delta":  np.array((  2, 129,  33),'f')/255.,
    "Theta":  np.array((  0,  76, 255),'f')/255.,
    "Lambda": np.array((118,   0, 136),'f')/255.,
    "Xi":     np.array((229,   0,   0),'f')/255.,
    "Pi":     np.array((255, 175, 199),'f')/255.,
    "Sigma":  np.array((115, 215, 238),'f')/255.,
    "Phi":    np.array((255, 141,   0),'f')/255.,
    "Psi":    np.array((255, 238,   0),'f')/255.
}
## draw Polygons Svg by drawsvg #####
import drawsvg

start = time()

def flattenPts(lst): # drowsvg
    return [item for sublist in lst for item in sublist] # drowsvg

SPECTRE_SHAPE = drawsvg.Lines(*flattenPts([p for p in SPECTRE_POINTS]), stroke="black", stroke_width=0.5,close=True) # drowsvg
Mystic_SPECTRE_SHAPE = drawsvg.Lines(*flattenPts([p for p in Mystic_SPECTRE_POINTS]), stroke="black",   stroke_width=0.5, close=True) # drowsvg

svgContens = drawsvg.Drawing(transformation_max - transformation_min,
                    transformation_max - transformation_min,
                     origin="center") # @TODO: ajust to polygons X-Y min and max. 
num_tiles = 0 # drowswvg

def drawPolygon2Svg(T, label): #drowsvg
    """
    T: transformation matrix
    label: label of shape type
    """
    global num_tiles,svgContens
    num_tiles += 1

    fill = f"rgb({int(round(COLOR_MAP[label][0]* 255, 0))}, {int(round(COLOR_MAP[label][1]* 255,0))}, {int(round(COLOR_MAP[label][2]* 255,0))})"
    stroke_f = "gray" # tile stroke color
    stroke_w = 0.1 if (fill[0] != 0) | (fill[1] != 0) | (fill[2] != 0) else 0 # tile stroke width
    shape = SPECTRE_SHAPE if label != "Gamma2" else Mystic_SPECTRE_SHAPE  # geometric points used.
    # print(f"transform-matrix,{T[0,0]},{T[1,0]},{T[0,1]},{T[1,1]},{T[0,2]},{T[1,2]}")

    svgContens.append(drawsvg.Use(
        shape,
        0, 0,
        transform=f"matrix({T[0,0]} {T[1,0]} {T[0,1]} {T[1,1]} {T[0,2]} {T[1,2]})",
        fill=fill,
        stroke=stroke_f,
        stroke_width=stroke_w))

tiles["Delta"].drawPolygon(drawPolygon2Svg) # updates num_tiles
saveFileName = f"spectre_tile{Edge_a:.1f}-{Edge_b:.1f}_{N_ITERATIONS}-{num_tiles}useRef.svg"
svgContens.save_svg(saveFileName)
time4 = time()-start
print(f"drowsvg: SVG drawing took {round(time4, 4)} seconds, generated {num_tiles} tiles")
print("drowsvg: drawPolygon save to " + saveFileName)
print(f"drowsvg: total processing time {round(time1+time4, 4)} seconds, {round(1000000*(time1+time4)/num_tiles, 4)} μs/tile")

# exit(0)

# draw Polygons Svg by matplotlib #####
# import matplotlib.pyplot as plt

# start = time()
# plt.figure(figsize=(8, 8))
# plt.axis('equal')

# num_tiles = 0
# def plotVertices(tile_transformation, label):
#     """
#     T: transformation matrix
#     label: label of shape type
#     """
#     global num_tiles
#     num_tiles += 1
#     vertices = (SPECTRE_POINTS if label != "Gamma2" else Mystic_SPECTRE_POINTS).dot(tile_transformation[:,:2].T) + tile_transformation[:,2]
#     # plt.text((vertices[1,0] + vertices[7,0])/2, (vertices[1,1] + vertices[7,1])/2, label, fontsize=8, color='gray')
#     plt.fill(vertices[:,0],vertices[:,1],facecolor=COLOR_MAP[label])
#     plt.plot(vertices[:,0],vertices[:,1],color='gray',linewidth=0.2)

# tiles["Delta"].drawPolygon(plotVertices)
# time2 = time()-start
# print(f"matplotlib.pyplot: tile recursion loop took {round(time2, 4)} seconds, generated {num_tiles} tiles")

# start = time()
# saveFileName = f"spectre_tile{Edge_a:.1f}-{Edge_b:.1f}_{N_ITERATIONS}-{num_tiles}pts.svg"
# print("matplotlib.pyplot: file save to " + saveFileName)
# plt.savefig(saveFileName)
# time3 = time()-start
# print(f"matplotlib.pyplot SVG drawing took {round(time3, 4)} seconds")
# print(f"matplotlib.pyplot total processing time {round(time1+time2+time3, 4)} seconds, {round(1000000*(time1+time2+time3)/num_tiles, 4)} μs/tile")

# plt.show()

