#!/usr/bin/env python

import vtk
import numpy as np

class VtkPointCloud:
    def __init__(self):
        self.vtkPolyData = vtk.vtkPolyData()
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.GetProperty().SetPointSize(5)
        self.callTimerCallbackFlag = True
        self.timeCount = 0
        self.pointCloud = []
        for k in xrange(100):
            self.pointCloud.append(20 * (np.random.rand(3) - 0.5))

    def addPoint(self, point):
        pointId = self.vtkPoints.InsertNextPoint(point[:])
        self.vtkDepth.InsertNextValue(point[2])
        self.vtkColor.InsertNextTupleValue([0,0.5,0])
        self.vtkCells.InsertNextCell(1)
        self.vtkCells.InsertCellPoint(pointId)

    def clearPointCloud(self):
        # point and cell
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        # depth
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        # color
        self.vtkColor = vtk.vtkUnsignedCharArray()
        self.vtkColor.SetNumberOfComponents(3)
        self.vtkColor.SetName('Colors')
        # poly
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')
        # self.vtkPolyData.GetPointData().SetScalars(self.vtkColor)
        # self.vtkPolyData.GetPointData().SetActiveScalars('Colors')

    def addPointCloud(self, points):
        self.clearPointCloud()
        for point in points:
            self.addPoint(point)

    def renderPointCloud(self, points):
        self.addPointCloud(points)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(self.vtkPolyData)
        self.vtkActor.SetMapper(mapper)
        self.renderer.AddActor(self.vtkActor)
        self.renderWindow.Render()

    def mainLoop(self):
        # Renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.vtkActor)
        self.renderer.SetBackground(1, 1, 1)
        self.renderer.ResetCamera()

        # Render Window
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.AddRenderer(self.renderer)

        # Interactor
        self.renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        self.renderWindowInteractor.SetRenderWindow(self.renderWindow)
        self.renderWindowInteractor.Initialize()

        # Set Callback
        self.renderWindowInteractor.AddObserver('KeyPressEvent', self.keyPressedCallback)
        self.renderWindowInteractor.AddObserver('TimerEvent', self.timerCallback)
        timerId = self.renderWindowInteractor.CreateRepeatingTimer(100);

        # Begin Interaction
        self.renderWindow.Render()
        self.initialCameraPose = dict()
        self.initialCameraPose['Position'] = self.renderer.GetActiveCamera().GetPosition()
        self.initialCameraPose['FocalPoint'] = self.renderer.GetActiveCamera().GetFocalPoint()
        self.initialCameraPose['ViewUp'] = self.renderer.GetActiveCamera().GetViewUp()
        # self.renderWindowInteractor.Render()
        self.renderWindowInteractor.Start()

    def timerCallback(self, obj, event):
        if self.callTimerCallbackFlag:
            self.timeCount += 1
            offset = [10 * np.sin(float(self.timeCount) / 10), 0, 0]
            tmpPointCloud = [point + offset for point in self.pointCloud]
            self.renderPointCloud(tmpPointCloud)

    def keyPressedCallback(self, obj, event):
        key = obj.GetKeySym()
        if key == 'r':
            self.renderer.GetActiveCamera().SetPosition(
                *self.initialCameraPose['Position'])
            self.renderer.GetActiveCamera().SetFocalPoint(
                *self.initialCameraPose['FocalPoint'])
            self.renderer.GetActiveCamera().SetViewUp(
                *self.initialCameraPose['ViewUp'])
        elif key == 's':
            self.callTimerCallbackFlag = not self.callTimerCallbackFlag

if __name__ == '__main__':
    pointCloud = VtkPointCloud()
    pointCloud.mainLoop()
