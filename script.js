const uploadInput = document.getElementById("imageUpload");
const preview = document.getElementById("preview");

uploadInput.onchange = evt => {
const [file] = uploadInput.files;
if (file) {
preview.src = URL.createObjectURL(file);
preview.style.display = "block";
}
};
