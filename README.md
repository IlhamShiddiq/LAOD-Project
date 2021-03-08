# LAOD
This is the final project of Intel AI for Youth Program. LAOD (Library Automatic Object Detection) is an AI that is used for data inputting about person who will borrow the book and which book that he/she will borrow. This is more efficient rather than inputting data manually. 

### Usage
* To add datasets, run the file named `faces-list.py`, fill the ID and the system will automatically open the camera and take the face photos of 250
* To train the datasets and create all of bottlenecks files, run `command 1` which is in the file named `commands-needed.txt`
* To test the model (CLI), run `command 2` which is in the file named `commands-needed.txt`<br/> note: fill `--image` argument with your image path
* To test the model (GUI), run `streamlit run app.py` and open link `http://localhost:8501/` on your browser

### Built With
* [Haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades) - Face detection
* [Tesseract](https://pypi.org/project/pytesseract/) - Text detection
* [Streamlit](https://streamlit.io/) - GUI
* [Tensorflow](https://www.tensorflow.org/)

### Contributors
* **Dafa Nurul Fauziansyah** - [Github](https://github.com/dafanf)
* **Ilham Shiddiq** - [Github](https://github.com/IlhamShiddiq)
