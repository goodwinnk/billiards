
## split_video_by_layout

Берет видео в формате `.mp4` и разметку ударов (`layout`),
режет видео на куски и сохраняет каждый кусок (удар) в отельный файл `.mp4`,
по умолчанию куски будут сохранены в том же месте, где находится исходное видео.

### Формат разметки

* Каждый удар в отдельной строчке

* Каждый в формате `start-finish`, где `start` и `finish` - это начальный и конечный момент
времени удара соответственно (в формате `h:m:s`, `h` или `h:m` может отсутстовать) 