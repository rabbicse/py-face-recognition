$(function () {
    console.log("Hello!");
});

$.ajax({
url: '/face/image/',
type: 'get',
success: function(data){
console.log("Hello   ..............1!");
console.log(data);}
});

