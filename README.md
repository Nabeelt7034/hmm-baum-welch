# Hidden Markov Model (HMM) – Baum Welch Algorithm

## 👨‍🎓 Student Details
- **Name:** Nabeel T
- **University Register Number:** LTCR24CS075

---

## 📌 Assignment Description

This assignment implements a **Hidden Markov Model (HMM)** using the **Baum–Welch Algorithm**.

The Baum–Welch algorithm is a training method used to estimate the parameters of an HMM when the hidden states are unknown.

Given:
- An observation sequence
- Number of hidden states

The program calculates:

- Transition probability matrix (A)
- Emission probability matrix (B)
- Initial state distribution (π)
- Log-likelihood P(O | λ) over iterations
- State transition diagram
- Log-likelihood convergence graph

This implementation is provided as a **localhost web application using Streamlit**.

---

## 🧠 What is Hidden Markov Model?

A Hidden Markov Model (HMM) is a statistical model used to represent systems that:

- Have hidden (unknown) states
- Produce observable outputs
- Follow probabilistic transitions

HMM is widely used in:

- Speech recognition
- Pattern recognition
- Bioinformatics
- Natural language processing

---

## ⚙️ How to Run the Program

### Step 1: Clone the Repository

```bash
git clone https://github.com/Nabeelt7034/hmm-baum-welch.git
cd hmm-baum-welch
