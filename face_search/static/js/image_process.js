$("#btnSubmit").click(function(event){
    //stop submit the form, we will post it manually.
    event.preventDefault();

    // Get form
    var form = $('#imgForm')[0];

    // Create an FormData object
    var data = new FormData(form);

    $.ajax({
        url: '/face/image/',
        type: 'POST',
        data: data,
        enctype: 'multipart/form-data',
        processData: false,  // Important!
        contentType: false,
        cache: false,
        timeout: 600000,
        success: function(response) {
            $("#uploaded_img").attr("src",response['image']);
            console.log(response);
        },
        error: function (e) {
            console.log("ERROR : ", e);
        }
    });
});

//function postAjaxRequest(e){
//    $.ajax({
//        url: '/face/image/',
//        type: 'get',
//        success: function(data){
//        console.log("Hello   ..............1!");
//        console.log(data);}
//    });
//}

