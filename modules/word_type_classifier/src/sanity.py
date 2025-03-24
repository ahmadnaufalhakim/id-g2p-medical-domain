from utils import preprocess_text

lst = [
  "--nya",
  "-anda",
  "-belas",
  "-compeng",
  "-delapan",
  "-kan",
  "-ku",
  "-lenggang",
  "-mahasiswa",
  "-nya",
  "-sepit",
  "-tak",
  "-wan",
  "-wati",
  "a'du",
  "a'had",
]
for el in lst :
  print(preprocess_text(el))