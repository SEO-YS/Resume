# Generated by Django 4.1.7 on 2023-05-09 11:51

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('datos', '0005_auto_20230501_0950'),
    ]

    operations = [
        migrations.AddField(
            model_name='question',
            name='content2',
            field=models.TextField(default=''),
        ),
        migrations.CreateModel(
            name='Question4',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('subject', models.CharField(max_length=200)),
                ('content', models.TextField()),
                ('content2', models.TextField(default='')),
                ('create_date', models.DateTimeField()),
                ('modify_date', models.DateTimeField(blank=True, null=True)),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='author_question4', to=settings.AUTH_USER_MODEL)),
                ('voter', models.ManyToManyField(related_name='voter_question4', to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Question3',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('subject', models.CharField(max_length=200)),
                ('content', models.TextField()),
                ('content2', models.TextField(default='')),
                ('create_date', models.DateTimeField()),
                ('modify_date', models.DateTimeField(blank=True, null=True)),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='author_question3', to=settings.AUTH_USER_MODEL)),
                ('voter', models.ManyToManyField(related_name='voter_question3', to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Question2',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('subject', models.CharField(max_length=200)),
                ('content', models.TextField()),
                ('content2', models.TextField(default='')),
                ('create_date', models.DateTimeField()),
                ('modify_date', models.DateTimeField(blank=True, null=True)),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='author_question2', to=settings.AUTH_USER_MODEL)),
                ('voter', models.ManyToManyField(related_name='voter_question2', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]