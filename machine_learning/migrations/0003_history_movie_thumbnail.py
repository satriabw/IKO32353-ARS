# -*- coding: utf-8 -*-
# Generated by Django 1.11.5 on 2017-11-17 14:24
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('machine_learning', '0002_history_confidence'),
    ]

    operations = [
        migrations.AddField(
            model_name='history',
            name='movie_thumbnail',
            field=models.URLField(null=True),
        ),
    ]
