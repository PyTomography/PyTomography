document.addEventListener('DOMContentLoaded', function () {
    let dropdown = document.getElementById('code-select');
    // Hide all code blocks
    let allCodeBlocks = document.querySelectorAll('.highlight-python');
    allCodeBlocks.forEach(el => el.style.display = 'none');
    // Add event listener for toggling
    dropdown.addEventListener('change', function () {
        let selectedValue = this.value;
        // Hide all code blocks
        let allCodeBlocks = document.querySelectorAll('.highlight-python');
        allCodeBlocks.forEach(el => el.style.display = 'none');
        if (document.getElementById(selectedValue)) {
            // Show the selected code block
            document.getElementById(selectedValue).style.display = '';
        }
    });
});