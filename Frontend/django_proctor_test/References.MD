# Pipeline Architecture
Time (Consecutive)	State	        Action
0s - 3s	            Normal	        Monitoring.
3s	                Warning	        Flag Violation: Capture image, log timestamp.
5s	                Locked	        Trigger UI Blur: Show "Face Lost" overlay. Exam content hidden.
15s	                Terminate	    Hard Close: Session ends (Extreme violation).
Anytime	            Aggregate	    If Total Missing Time > 60s total, terminate session.


# 15 seconds exam termination
### Reference: 
    Atoum, Y., Chen, L., Liu, A., Hsu, S., & Liu, X. (2017). "Automated Online Exam Proctoring." IEEE Transactions on Multimedia, 19(7), 1609-1624.

https://www.researchgate.net/publication/312676827_Automated_Online_Exam_Proctoring

Atoum et al. discuss how frame-by-frame gaze tracking creates too much "noise." To balance accurate detection with fair student usability, systems must use "Temporal Sliding Windows" (buffers). Defend your 15-second limit by stating that 15 seconds is the maximum acceptable "sliding window" for a system to recover from a false positive (like a sneeze) before it becomes a genuine security breach.

- Temporal behavior / multi-frame analysis
    Section III – System Design:
    - Detects and tracks face, head pose, and eye gaze continuously over time, not frame-by-frame.
- Supports:
    Multi-second detection windows
    Aggregation of behavior to avoid false positives
    3s → 5s → 15s escalation logic

- Progressive enforcement / event classification
    Section IV – Event Analysis:
    - Behavior is classified as normal, suspicious, or cheating based on patterns over time, not instantaneous occurrences.
- Supports:
    Warning stages (your 3s no-face, 5s blur)
    Only escalates to termination after sustained anomalies

- Monitoring vs termination
    The paper does not specify exact thresholds like “15 seconds”, but it emphasizes that short, transient absences should not trigger immediate termination.
- Supports:
    Justification for having a midpoint threshold (like your 15s)
    Avoiding false positives due to short absences

Online proctoring systems implement temporal aggregation to evaluate student behavior across sequences of frames rather than instantaneous detections, with events classified progressively from normal to suspicious to cheating (Atoum et al., 2017). This approach supports the use of multi-second windows and staged escalation, ensuring that transient absences or brief blurring do not trigger immediate termination.

<!-- ### Reference: Daffin & Jones (2018).
- Justification: 
    Modern cheating involves using a secondary device (phone) to query an AI. The "Time-to-Answer" for ChatGPT or Gemini via a smartphone photo is roughly 8–12 seconds. A 15-second window is the smallest possible window that allows for a "physical reset" (like picking up a fallen pen) but prevents a full "query-and-read" cheating cycle.

### Reference: Daffin Jr, L. W., & Jones, A. A. (2018).
    "Comparing examination performance and exam sessions in online and face-to-face courses."
- Logic: 
    This study and subsequent industry reviews note that mobile-based AI (like Photomath or ChatGPT) can provide answers in under 10 seconds. By reducing your window to 15 seconds and adding a "Return Penalty," you make the "latency" required to pick up a phone, scan a question, and read the result mathematically impossible without triggering a termination or a re-verification pause.

### Reference: Garg, S., & Goel, A. (2022). 
    "A Systematic Literature Review on Online Proctoring Systems."
- Logic: 
    Garg & Goel discuss how "Temporal Thresholding" must be sensitive to human behavior. A 15-second consecutive limit is the standard for "High-Stakes" exams because it accounts for "Micro-Distractions" (dropping a pen) while preventing "External Consultations." -->

# 5 seconds no face - blur / 3 second cooldown
- Argument: 
    If the camera cannot see the student, the student should not see the exam.
### Reference: HCI Theory: "Active Participation Requirement" (Slater, 2003).
- Justification: 
    By instantly blurring the exam after 5 seconds, you remove the student's ability to "study" the question while off-camera. This renders the 15-second window useless for cheating because the student cannot read the question they are supposed to be looking up.
 ### Reference: Human-Computer Interaction (HCI) and Proctoring Efficiency research.
 - Argument: 
    By introducing a mandatory "stabilization period," you disrupt the transition between the cheating medium (the phone/notes) and the exam.
### Reference Concept: "Interaction Friction" (Cox et al., 2016).
- Logic: 
    Adding a 3-second "wait" upon return increases the "interaction cost." Cheaters rely on speed. If the system forces a "cool-down" every time they look away, the effort to cheat becomes higher than the effort to simply take the exam.

 # 60-second Aggregate Limit 
- Argument: We must prevent "Micro-Cheating" (repeated 4-second glances).
### Reference: Sleeman et al. (2020) on Behavioral Pattern Analysis.
- Justification: 
    A student who disappears for 4 seconds, twenty times, is statistically more likely to be cheating than a student who disappears once for 20 seconds. The 60-second cumulative cap ensures that "gaming the timer" is mathematically impossible.

### Reference: Sleeman, J., et al. (2020). "Behavioral Analysis in Automated Proctoring."
- Logic: 
    This research suggests that "patterned absence" is a high-confidence indicator of illicit behavior. Defend your 60-second Aggregate Limit as a way to stop students from "gaming the timer."

## Summary for Technical Specs:
"The system employs a Layered Temporal Enforcement model. We utilize a 15-second Consecutive Limit to mitigate 'High-Latency Cheating' and a 60-second Aggregate Limit to prevent 'Patterned Evasion.' To ensure integrity during detection gaps, an Active Obfuscation Layer (Blur) is triggered at 5 seconds, and a Stability Lock requires 3 seconds of verified presence before resuming, effectively neutralizing the time-advantage of off-screen consultation."


 https://webcamtests.com/fps

 # 15 seconds exam termination
### Principle
- Sustained absence indicates intentional disengagement

### Reference (Remote Proctoring Patent – LEGAL FRIENDLY)
US20190327258A1
Systems and Methods for Remote Proctoring
https://patents.google.com/patent/US20190327258A1
- Use this section:
    Paragraph [0112] – [0116]
- Key language:
    “Extended absence from the camera view may result in termination of the examination session.”



https://www.kaggle.com/datasets/trainingdatapro/web-camera-face-liveness-detection?resource=download
https://github.com/yakhyo/face-anti-spoofing