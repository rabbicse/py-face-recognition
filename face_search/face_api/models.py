from django.db import models


# Create your models here.
class Image(models.Model):
    image = models.FileField(blank=False, null=False)

    def __str__(self):
        return self.file.name
