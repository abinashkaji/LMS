# Docker for Django Project

This demo repository provides a Docker-based setup for running a Django project LMS (library Management System). Docker allows you to create isolated environments, ensuring consistent development and deployment across different systems.

## Prerequisites

Before you begin, make sure you have the following installed on your system:
- Docker: [Install Docker](https://www.docker.com/get-started)

## Getting Started

Follow these steps to set up and run your Django project using Docker:

1. **Clone the Repository:**
   ```
   git clone [https://github.com/abinashkaji/demo.git]
   cd django-docker-project
   ```

2. **Build the Docker Image:**
   ```
   docker build -t demo .
   ```

3. **Database Configuration (Optional):**
   If you need to configure the database settings, modify the `DATABASES` dictionary in the `settings.py` file of your Django project.

4. **Start the Docker Containers:**
   ```
   docker-compose up
   ```

5. **Create Django Superuser :**
   If you are using Django's admin interface, you can create a superuser using the following command:
   ```
   docker-compose run web python manage.py createsuperuser
   ```

6. **Access Your Django Application:**
   Your Django application should now be running. Open your web browser and go to `http://localhost:8000` to access the application. If you created a superuser, you can access the admin interface at `http://localhost:8000/admin`.

7. **Shut Down the Containers:**
   To stop and remove the running containers, use the following command:
   ```
   docker-compose down
   ```

## Customizing the Docker Configuration

The Docker configuration is defined in the `Dockerfile` and `docker-compose.yml` files. If your Django project requires additional dependencies or services, you can modify these files accordingly.

## Directory Structure

This repository has the following structure:

```
demo/
  ├── LMS/
  │   ├── manage.py
  │   ├── demo/
  │   │   ├── __init__.py
  │   │   ├── settings.py
  │   │   ├── urls.py
  │   │   └── ...
  │   └── ...
  ├── Dockerfile
  ├── LICENSE
  ├── requirements.txt
  └── ...
```

- The `LMS` directory contains your Django project files for Library management system.
- The `Dockerfile` specifies how to build the Docker image for your Django application.
- The `docker-compose.yml` can be define for the services, including the Django web server and database in the demo directory.

## Notes

- For development, Django's `DEBUG` setting is set to `True`. Change it to `False` for production deployments.
- You can customize the Docker setup to include additional services like caching, load balancing, etc., based on your project requirements.

## Contributing

If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

Happy coding! 🚀
