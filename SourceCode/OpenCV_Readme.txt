This is instruction on setting up openCV 3.1.0 on Visual Studio 2015 in Windows 10 64 bit OS
1. Download and install OpenCV 3.1.0 from http://opencv.org/downloads.html
2. Add OpenCV to environmental path
   System variable:
	Variable name: OPENCV_BUILD
	Variable Value: <Installation Path>\opencv\build
   Path: append ";%OPENCV_BUILD%\x64\vc14\bin"
3. In Visual Studio, create project use existing code
4. Open project property
5. Common Properties -> C/C++ -> Additional Include Directories, add "$(OPENCV_BUILD)\include"
6. Common Properties -> Linker -> General -> Additional Library Directories, add "$(OPENCV_BUILD)\x64\vc14\lib" 
7. Common Properties -> Linker -> Input -> Additional Dependencies, add "opencv_world310d.lib"
