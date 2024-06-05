
const myForm = document.getElementById('myForm');
const inpFile = document.getElementById('inpFile');

// this will prevent refreshing or redirecting the page
myForm.addEventListener('submit', e => {
    e.preventDefault();

    // define endpoint where file will be stored
    const endpoint = "../../templates/upload.php";
    const formData = new FormData();

    // to print the logs
    console.log(inpFile.files);

    // we are fethching file name of index 0 and appneding it
    formData.append("inpFile", inpFile.files[0]);

    fetch(endpoint, {
        method: "post",
        body: formData
    }).catch(console.error);

});


