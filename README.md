<div align="center">
  
# ClearPath-AI
### A Vision-Based Intelligent Traffic Signal Control System for Emergency Vehicle Prioritization

</div>

---

## Abstract

ClearPath-AI is a research-oriented project that explores the use of **computer vision and real-time decision systems** to automatically prioritize emergency vehicles at traffic intersections.

The system detects emergency vehicles using a deep learning object detection model and dynamically controls traffic signals to create a **green corridor**, reducing response time and improving the effectiveness of emergency services in urban environments.

---

## Motivation

Urban traffic congestion is a major factor contributing to **delays in emergency response**.  
Traditional traffic signal systems are static or semi-adaptive and **do not account for real-time emergency scenarios**.

Even small delays can have **life-critical consequences** in medical and disaster-response situations.

This project is motivated by the need for an **intelligent, automated, and vision-based traffic control mechanism** that can react to emergency vehicles in real time without relying on manual intervention or special hardware in vehicles.

---

## Problem Statement

To design a system that:

- Can detect emergency vehicles using roadside camera feeds  
- Can make real-time decisions at traffic intersections  
- Can dynamically override traffic signals to prioritize emergency vehicles  
- While remaining scalable and deployable in smart-city environments  

---

## Proposed Approach

The proposed system uses:

- A vision-based object detection model (YOLOv8) for identifying emergency vehicles  
- A rule-based decision engine for traffic signal control  
- A real-time video processing pipeline to ensure low-latency response  

The system temporarily alters the signal phase of the intersection in favor of the detected emergency vehicle and restores normal operation after it passes.

---

## System Architecture

```text
Traffic Camera
      ↓
YOLOv8 Detection Model
      ↓
Emergency Vehicle Classification
      ↓
Decision & Control Logic
      ↓
Traffic Signal Controller
