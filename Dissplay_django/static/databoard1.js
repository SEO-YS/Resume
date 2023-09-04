
const card= document.getElementById('myCard');

card.addEventListener('mouseover', addShadow);
card.addEventListener('mouseout', removeShadow);

function addShadow() {
  card.style.boxShadow = '0px 0px 20px rgba(0, 0, 0, 0.9)';
}

function removeShadow() {
  card.style.boxShadow = '0px 0px 10px rgba(0, 0, 0, 0)';
}



card.addEventListener('mouseenter', liftImage);
card.addEventListener('mouseleave', resetImage);

function liftImage() {
  card.style.transform = 'translateY(-10px)';
}

function resetImage() {
  card.style.transform = 'translateY(0)';
}
