import numpy as np
import math
import copy


class node():
    def __init__(self, parent, ind, leftChild, rightChild, faces, fLength, value, dim):
        self.parent = parent
        self.ind = ind
        self.value = value
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.facesLeft = []
        self.numLeft = 0
        self.facesRight = []
        self.numRight = 0
        self.dim = dim
        for i in range(fLength):
            if(faces[i].vert0[dim] < value and faces[i].vert1[dim] < value and faces[i].vert2[dim] < value):
                self.facesLeft.append(faces[i])
                self.numLeft += 1
            elif(faces[i].vert0[dim] >= value and faces[i].vert1[dim] >= value and faces[i].vert2[dim] >= value):
                self.numRight += 1
                self.facesRight.append(faces[i])
            else:
                self.numLeft += 1
                self.numRight += 1
                self.facesLeft.append(faces[i])
                self.facesRight.append(faces[i])


class face():
    def __init__(self, x, y, z):
        self.vert0 = [x[0], y[0], z[0]]
        self.vert1 = [x[1], y[1], z[1]]
        self.vert2 = [x[2], y[2], z[2]]

class kdTree():
    def __init__(self, x, y, z):
        self.faces = []
        for i in range(x.shape[0]):
            self.faces.append(face(x[i,:], y[i,:], z[i,:]))

        #save raw data
        self.points = np.empty((3*x.shape[0], x.shape[1]))
        self.points[:,0] = np.concatenate((x[:,0],x[:,1],x[:,2]), axis=0)
        self.points[:,1] = np.concatenate((y[:,0],y[:,1],y[:,2]), axis=0)
        self.points[:,2] = np.concatenate((z[:,0],z[:,1],z[:,2]), axis=0)
        self.nodes = [None] * (2**6 -1)

        self.buildTree(0,self.points,0,-1)

    def buildTree(self,level,points,index,parent):
        dim = level % 3
        points = points[points[:,dim].argsort(), :]
        medInd = points.shape[0] // 2
        median = points[medInd, dim]
        if not level:
            self.nodes[0] = node(-1, 0, 1, 2, self.faces, len(self.faces), median, 0)
        else: 
            if index % 2:
                facesbin = copy.deepcopy(self.nodes[parent].facesLeft)
            else:
                facesbin = copy.deepcopy(self.nodes[parent].facesRight)
                self.nodes[parent].facesLeft = None
                self.nodes[parent].facesRight = None
            if level < 5:
                self.nodes[index] = node(parent, index, index*2+1, index*2+2, facesbin, len(facesbin), median, dim)
            else:
                self.nodes[index] = node(parent, index, None, None, facesbin, len(facesbin), median, dim)


        if level < 5:
            current = self.nodes[index]
            pointsLeft = np.empty((current.numLeft*3, 3))
            pointsRight = np.empty((current.numRight*3, 3))
            if current.numLeft > current.numRight:
                inds = current.numLeft
            else:
                inds = current.numRight
            for i in range(inds):
                if i < current.numLeft:
                    pointsLeft[i*3,:] = current.facesLeft[i].vert0
                    pointsLeft[i*3+1,:] = current.facesLeft[i].vert1
                    pointsLeft[i*3+2,:] = current.facesLeft[i].vert2
                elif i < current.numRight:
                    pointsRight[i*3,:] = current.facesRight[i].vert0
                    pointsRight[i*3+1,:] = current.facesRight[i].vert1
                    pointsRight[i*3+2,:] = current.facesRight[i].vert2

            self.buildTree(level+1, pointsLeft, index*2+1, index)
            self.buildTree(level+1, pointsRight, index*2+2, index)
        else:
            print(self.nodes[index].numLeft, self.nodes[index].numRight)
        return

    def pointTriangleDistance(self, TRI, P):
        # function [dist,PP0] = pointTriangleDistance(TRI,P)
        # calculate distance between a point and a triangle in 3D
        # SYNTAX
        #   dist = pointTriangleDistance(TRI,P)
        #   [dist,PP0] = pointTriangleDistance(TRI,P)
        #
        # DESCRIPTION
        #   Calculate the distance of a given point P from a triangle TRI.
        #   Point P is a row vector of the form 1x3. The triangle is a matrix
        #   formed by three rows of points TRI = [P1;P2;P3] each of size 1x3.
        #   dist = pointTriangleDistance(TRI,P) returns the distance of the point P
        #   to the triangle TRI.
        #   [dist,PP0] = pointTriangleDistance(TRI,P) additionally returns the
        #   closest point PP0 to P on the triangle TRI.
        #
        # Author: Gwolyn Fischer
        # Release: 1.0
        # Release date: 09/02/02

        # rewrite triangle in normal form
        B = TRI[0, :]
        E0 = TRI[1, :] - B
        # E0 = E0/sqrt(sum(E0.^2)); %normalize vector
        E1 = TRI[2, :] - B
        # E1 = E1/sqrt(sum(E1.^2)); %normalize vector
        D = B - P
        a = np.dot(E0, E0)
        b = np.dot(E0, E1)
        c = np.dot(E1, E1)
        d = np.dot(E0, D)
        e = np.dot(E1, D)
        f = np.dot(D, D)

        #print "{0} {1} {2} ".format(B,E1,E0)
        det = a * c - b * b
        s = b * e - c * d
        t = b * d - a * e

        # Terible tree of conditionals to determine in which region of the diagram
        # shown above the projection of the point into the triangle-plane lies.
        if (s + t) <= det:
            if s < 0.0:
                if t < 0.0:
                    # region4
                    if d < 0:
                        t = 0.0
                        if -d >= a:
                            s = 1.0
                            sqrdistance = a + 2.0 * d + f
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
                    else:
                        s = 0.0
                        if e >= 0.0:
                            t = 0.0
                            sqrdistance = f
                        else:
                            if -e >= c:
                                t = 1.0
                                sqrdistance = c + 2.0 * e + f
                            else:
                                t = -e / c
                                sqrdistance = e * t + f

                                # of region 4
                else:
                    # region 3
                    s = 0
                    if e >= 0:
                        t = 0
                        sqrdistance = f
                    else:
                        if -e >= c:
                            t = 1
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f
                            # of region 3
            else:
                if t < 0:
                    # region 5
                    t = 0
                    if d >= 0:
                        s = 0
                        sqrdistance = f
                    else:
                        if -d >= a:
                            s = 1
                            sqrdistance = a + 2.0 * d + f;  # GF 20101013 fixed typo d*s ->2*d
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
                else:
                    # region 0
                    invDet = 1.0 / det
                    s = s * invDet
                    t = t * invDet
                    sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
        else:
            if s < 0.0:
                # region 2
                tmp0 = b + d
                tmp1 = c + e
                if tmp1 > tmp0:  # minimum on edge s+t=1
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        s = 1.0
                        t = 0.0
                        sqrdistance = a + 2.0 * d + f;  # GF 20101014 fixed typo 2*b -> 2*d
                    else:
                        s = numer / denom
                        t = 1 - s
                        sqrdistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f

                else:  # minimum on edge s=0
                    s = 0.0
                    if tmp1 <= 0.0:
                        t = 1
                        sqrdistance = c + 2.0 * e + f
                    else:
                        if e >= 0.0:
                            t = 0.0
                            sqrdistance = f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f
                            # of region 2
            else:
                if t < 0.0:
                    # region6
                    tmp0 = b + e
                    tmp1 = a + d
                    if tmp1 > tmp0:
                        numer = tmp1 - tmp0
                        denom = a - 2.0 * b + c
                        if numer >= denom:
                            t = 1.0
                            s = 0
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = numer / denom
                            s = 1 - t
                            sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

                    else:
                        t = 0.0
                        if tmp1 <= 0.0:
                            s = 1
                            sqrdistance = a + 2.0 * d + f
                        else:
                            if d >= 0.0:
                                s = 0.0
                                sqrdistance = f
                            else:
                                s = -d / a
                                sqrdistance = d * s + f
                else:
                    # region 1
                    numer = c + e - b - d
                    if numer <= 0:
                        s = 0.0
                        t = 1.0
                        sqrdistance = c + 2.0 * e + f
                    else:
                        denom = a - 2.0 * b + c
                        if numer >= denom:
                            s = 1.0
                            t = 0.0
                            sqrdistance = a + 2.0 * d + f
                        else:
                            s = numer / denom
                            t = 1 - s
                            sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

        # account for numerical round-off error
        if sqrdistance < 0:
            sqrdistance = 0

        PP0 = B + s * E0 + t * E1
        return sqrdistance, PP0

    def checkFaces(self, faces, point, clPoint, dist):
        triangle = np.empty((3,3))
        for face in faces:
            triangle[0,:] = face.vert0
            triangle[1,:] = face.vert1
            triangle[2,:] = face.vert2
            tmpDist, tmpPoint = self.pointTriangleDistance(triangle, point)
            if tmpDist < dist:
                dist = tmpDist
                clPoint = tmpPoint
        return dist, clPoint


    def traverse(self, ind, point, clPoint, dist):
        current = self.nodes[ind]
        while 1:
            if(point[current.dim] < current.value):
                if current.leftChild is None:
                    break
                current = self.nodes[current.leftChild]
            else:
                if current.rightChild is None:
                    break
                current = self.nodes[current.rightChild]

        check = True
        if(point[current.dim] < current.value):
            faces = current.facesLeft[:]
            if not len(faces):
                check = False
                faces = current.facesRight[:]
            left = True
        else:
            faces = current.facesRight[:]
            if not len(faces):
                check = False
                faces = current.facesLeft[:]
            left = False
        dist, clPoint = self.checkFaces(faces, point, clPoint, dist)
        distPlane = (point[current.dim] - current.value)**2
        if distPlane < dist and check:
            if(left):
                faces = current.facesRight[:]
                if not len(faces):
                    return clPoint, dist, current.ind
                dist, clPoint = self.checkFaces(faces, point, clPoint, dist)
            else:
                faces = current.facesLeft[:]
                if not len(faces):
                    return clPoint, dist, current.ind
                dist, clPoint = self.checkFaces(faces, point, clPoint, dist)

        return clPoint, dist, current.ind


    def kd_search(self, point):
        current = self.nodes[0]
        clPoint, dist, ind = self.traverse(0, point, [0,0,0], 100)
        current = self.nodes[ind]
        checked = [0] *6
        checked[0] = ind
        counter = 0
        while current.parent > -1:
            goLeft = current.ind == self.nodes[current.parent].rightChild
            current = self.nodes[current.parent]
            counter += 1
            if checked[counter] is not current.ind:
                distPlane = (point[current.dim] - current.value)**2
                if distPlane < dist:
                    checked[counter] = current.ind
                    counter = 0
                    if goLeft:
                        clPoint, dist, ind = self.traverse(self.nodes[current.leftChild].ind, point, clPoint, dist)
                    else:
                        clPoint, dist, ind = self.traverse(self.nodes[current.rightChild].ind, point, clPoint, dist)
                    current = self.nodes[ind]

        dist = math.sqrt(dist)
        return dist, clPoint

    def query(self, points):
        dist = np.empty((points.shape[0], 1))
        closestPts = np.empty((points.shape))
        for i in range(points.shape[0]):
            dist[i], closestPts[i,:] = self.kd_search(points[i,:])
        return dist, closestPts


