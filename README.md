<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"
    integrity="sha512-KfkfwYDsLkIlwQp6LFnl8zNdLGxu9YAA1QvwINks4PhcElQSvqcyVLLD9aMhXd13uQjoXtEKNosOWaZqXgel0g=="
    crossorigin="anonymous" referrerpolicy="no-referrer" />
<link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@400;700&display=swap" rel="stylesheet">

<div align="center" style="font-family: 'QuickSand';">
    <h2 style="font-family: 48px;">Face Detection and Tracking Drone</h2>
    <p style="font-family: 24px;">My Final Year Project on Autonomous Face Tracking for DJI Tello</p>
    <p align="center">
        <img src="https://img.shields.io/github/languages/count/sakthyvell/Face-Detection-and-Tracking-UAV" alt="">
        <img src="https://img.shields.io/github/languages/top/sakthyvell/Face-Detection-and-Tracking-UAV" alt="">
        <img src="https://img.shields.io/github/last-commit/sakthyvell/Face-Detection-and-Tracking-UAV" alt="">
        <img src="https://img.shields.io/badge/development-completed-blue" alt="">
    </p>
</div>

<hr>
<br>
The project uses Harr-Cascaade Classifier to detect human faces. Using the bounding boxes drawn on the face, the controls are mapped to the drone to maintain itself in the center of the bounding box.

### Installation
#### Recommended : Install virtualenv
```bash
$ pip install virtualenv
$ virtualenv venv
```
#### Install dependencies
```bash
$ pip install -r requirements.txt
```

### Running Flask App
To run the flask app
```bash
$ python TelloTV.py
```
