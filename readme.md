<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

# 킥라니 멈춰!

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

<!-- TABLE OF CONTENTS -->

<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
        <ul>
        <li><a href="">Used Deivce</a></li>
       </ul>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
       <ul>
        <li><a href="">Architecture</a></li>
       </ul>
    </li>
    <li>
      <a href="#about-the-project">Trained Indicators</a>
    </li>  
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage example</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>






<!-- ABOUT THE PROJECT -->

## About The Project

This is the project detecting illegal electric scooter riders on the public street with unmanned device. And this **repository** contains the 2nd part of the detection (number of person/helmet, kickboard brand). It receives the rider data from **jetson nano** and detects the object with **YOLOv5** model and sends the data to Database.

![](/banner.jpg)

### Used Device

- **AWS EC2 with GPU**
- GPU : NVIDIA M60 8GB


### Built With

* [Python 3.8.10](https://www.python.org/)

* [Django 3.2.3](https://www.djangoproject.com/)

* [pytorch 1.8.1](https://pytorch.org/)

* [YOLOv5](https://github.com/ultralytics/yolov5)

* [MQTT](https://mqtt.org/)

### Architecture

![](https://user-images.githubusercontent.com/45448869/120573177-201efd00-c458-11eb-9e45-4a5b39421264.png)

## Trained Indicators

| Model | number of dataset | mAP<sup>val<br>0.5 | mAP<sup>test<br>0.5|
| ----------------- | --------------------- | ----------------------- | ------------------------ |
| Rider (x) | 983 | 0.981 | 0.955 |
| Kickboard (m)| 1500| 0.94 | 0.843 |
| Person (s) | 712 | 0.992 | 0.978 |
| helmet (m) | 629 | 0.984 | - |

![](https://user-images.githubusercontent.com/45448869/120500749-0b148080-c3fc-11eb-9220-14953a88b572.gif)

![](https://user-images.githubusercontent.com/45448869/120500745-09e35380-c3fc-11eb-96d0-838e13a10c13.gif)

![](https://user-images.githubusercontent.com/45448869/120500741-08b22680-c3fc-11eb-88f3-ce37c9f6906e.gif)

![](https://user-images.githubusercontent.com/45448869/120592863-823c2a00-c479-11eb-86ac-d953b6b64ae1.gif)

<!-- GETTING STARTED -->

## Getting Started

To get a local copy up and running follow these simple steps.

### Installation

1. Clone the repo

   ```shell
   git clone https://github.com/bururaru/stop_kickrani_BE.git
   ```

2. Install libraries with `requirements.txt`

   ```shell
   pip install -r requirements.txt
   ```

3. Run the Django server

   

<!-- USAGE EXAMPLES -->

## Usage example

 [Dashboard](https://example.com)







<!-- ROADMAP -->

## Roadmap

See the [open issues](https://github.com/bururaru/stop_kickrani_BE/issues) for a list of proposed features (and known issues).



<!-- CONTACT -->

## Contact

김기범 - [kimgibum90@gmail.com](mailto:kimgibum90@gmail.com)

이상락 - [leesangrak0307@gmail.com](mailto:leesangrak0307@gmail.com)

이창민 - bururaru@gmail.com

장소영 - [so970404@gmail.com](mailto:so970404@gmail.com)

정수경 - [skjung247@gmail.com](mailto:skjung247@gmail.com)

Project Link: [https://github.com/bururaru/stop_kickrani_BE](https://github.com/bururaru/stop_kickrani_BE)







<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/bururaru/stop_kickrani_BE.svg?style=for-the-badge
[contributors-url]: https://github.com/bururaru/stop_kickrani_BE/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/bururaru/stop_kickrani_BE.svg?style=for-the-badge
[forks-url]: https://github.com/bururaru/stop_kickrani_BE/network/members
[stars-shield]: https://img.shields.io/github/stars/bururaru/stop_kickrani_BE.svg?style=for-the-badge
[stars-url]: https://github.com/bururaru/stop_kickrani_BE/stargazers
[issues-shield]: https://img.shields.io/github/issues/bururaru/stop_kickrani_BE.svg?style=for-the-badge
[issues-url]: https://github.com/bururaru/stop_kickrani_BE/issues
