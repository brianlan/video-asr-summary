```
WHISPER vs PYANNOTE SEGMENTATION: A Visual Comparison
=====================================================

Example Timeline (10 seconds of audio with 2 speakers):
Time:    0    1    2    3    4    5    6    7    8    9   10
         |    |    |    |    |    |    |    |    |    |    |

🎤 WHISPER SEGMENTATION (Content-Based):
         [-------- "Hello, how are you?" --------]
                           [------- "I'm doing well, thanks" -------]
                                                    [-- "Great!" --]

🗣️ PYANNOTE SEGMENTATION (Speaker-Based):
         [------- Speaker_A -------]  [------- Speaker_B -------]
                                                 [- Speaker_A -]

🔗 INTEGRATED RESULT:
         Speaker_A: "Hello, how are you?"     [0.0-3.5]
         Speaker_B: "I'm doing well, thanks"  [3.5-7.8] 
         Speaker_A: "Great!"                  [8.0-9.2]

KEY INSIGHTS:
============

1. DIFFERENT PURPOSES:
   • Whisper: Captures natural speech chunks (sentences, phrases)
   • Pyannote: Captures speaker turns (who is talking when)

2. DIFFERENT BOUNDARIES:
   • Whisper boundaries follow speech content and natural pauses
   • Pyannote boundaries follow voice changes between speakers
   • They rarely align perfectly!

3. INTEGRATION CHALLENGE:
   • Whisper segment [0.0-3.5]: "Hello, how are you?"
   • Pyannote segment [0.0-3.2]: Speaker_A
   • Overlap: 3.2/3.5 = 91% → Assign to Speaker_A ✓

4. WHY BOTH ARE NEEDED:
   • Whisper alone: Great transcription, no speaker info
   • Pyannote alone: Speaker timing, no content
   • Together: Rich speaker-attributed transcripts

REAL-WORLD EXAMPLE (from our Chinese audio):
==========================================

Whisper Output:
[0.00-29.98] "第一章,行为,The Behaviour,我们已经准备好合适的策略了..."
[30.00-58.28] "在前几小时到几天内,是什么改变了神经系统对某个刺激的敏感度?..."

Pyannote Output (hypothetical):
[0.00-45.5] Speaker_00
[45.5-60.0] Speaker_00  (same speaker, but pyannote detected slight voice change)

Integration Result:
Speaker_00: "第一章,行为,The Behaviour,我们已经准备好合适的策略了..." [0.00-29.98]
Speaker_00: "在前几小时到几天内,是什么改变了神经系统对某个刺激的敏感度?..." [30.00-58.28]

This is why both segmentation types are essential for comprehensive 
speaker-attributed transcription!
```
