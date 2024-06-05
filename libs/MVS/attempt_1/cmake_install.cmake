# Install script for directory: D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/openMVS")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/make/Debug/MVS.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/make/Release/MVS.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/make/MinSizeRel/MVS.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/make/RelWithDebInfo/MVS.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/MVS/Camera.h;/MVS/Common.h;/MVS/DepthMap.h;/MVS/Image.h;/MVS/Interface.h;/MVS/Mesh.h;/MVS/PatchMatchCUDA.h;/MVS/PatchMatchCUDA.inl;/MVS/Platform.h;/MVS/PointCloud.h;/MVS/RectsBinPack.h;/MVS/Scene.h;/MVS/SceneDensify.h;/MVS/SemiGlobalMatcher.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/MVS" TYPE FILE FILES
    "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/Camera.h"
    "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/Common.h"
    "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/DepthMap.h"
    "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/Image.h"
    "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/Interface.h"
    "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/Mesh.h"
    "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/PatchMatchCUDA.h"
    "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/PatchMatchCUDA.inl"
    "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/Platform.h"
    "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/PointCloud.h"
    "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/RectsBinPack.h"
    "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/Scene.h"
    "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/SceneDensify.h"
    "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/SemiGlobalMatcher.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "D:/BlueMarble/TestZone/openMVS-bmg/libs/MVS/make/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
