// Script for Smart To-Do List application

document.addEventListener('DOMContentLoaded', () => {
    const taskInput = document.getElementById('taskInput');
    const addTaskButton = document.getElementById('addTaskButton');
    const tasksContainer = document.getElementById('tasksContainer');

    addTaskButton.addEventListener('click', () => {
        const taskValue = taskInput.value.trim();
        if (taskValue) {
            const taskElement = document.createElement('div');
            taskElement.classList.add('task-item');
            taskElement.innerHTML = `
                <span class='task-text'>${taskValue}</span>
                <button class='btn-delete'>Delete</button>
            `;
            tasksContainer.appendChild(taskElement);
            taskInput.value = '';

            taskElement.querySelector('.btn-delete').addEventListener('click', () => {
                tasksContainer.removeChild(taskElement);
            });
        }
    });
});