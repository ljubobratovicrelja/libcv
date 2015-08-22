// The MIT License (MIT)
// 
// Copyright (c) 2015 Relja Ljubobratovic, ljubobratovic.relja@gmail.com
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#ifndef CV_IGNORE_GUI // ignore compilation of the source file.


#include "../include/gui.hpp"
#include "../include/array.hpp"

#ifdef WIN32
// TODO: integrate qt include files in thirdParty directory.
#include <QtWidgets/QDialog>
#include <QtWidgets/QKeyEvent>
#include <QtWidgets/QCloseEvent>
#include <QtWidgets/QLabel>
#include <QtWidgets/QApplication>
#include <QtWidgets/QBoxLayout>
#else
#include <QtGui/QDialog>
#include <QtGui/QKeyEvent>
#include <QtGui/QCloseEvent>
#include <QtGui/QLabel>
#include <QtGui/QBoxLayout>
#include <QtGui/QApplication>
#endif

namespace cv {

class CV_EXPORT image_window : public QDialog {
	Q_OBJECT
private:
	QLabel *image_lbl;
public:
	explicit image_window(QWidget *parent = nullptr);
	~image_window();

	void set_image(const image_array &image);

	virtual void keyPressEvent(QKeyEvent *e);
	virtual void closeEvent(QCloseEvent *c);
};

class CV_EXPORT global_image_application: public QApplication {
private:
	static global_image_application* _singleton;
	std::vector<image_window*> _windows;
	bool _force_quit;
	char _last_key;

	global_image_application(int &argc, char** argv);
public:
	~global_image_application();

	static global_image_application *singleton();

	void force_quit();
	void assign_key(char key);
	char get_last_key() const;

	void create_window(const std::string &name, const image_array &image);
	void destroy_window(const std::string &name);
	void destroy_all_windows();
};

image_window::image_window(QWidget *parent): QDialog(parent) {
	QHBoxLayout *lay = new QHBoxLayout(this);
	lay->setContentsMargins(QMargins(0, 0, 0, 0));
	image_lbl = new QLabel("", this);
	lay->addWidget(image_lbl);
	this->setLayout(lay);
	this->resize(500, 500);
}

image_window::~image_window() {

}

void image_window::set_image(const image_array &image) {
	if (!image.is_valid()) {
		return;
	}
	QImage::Format format;
	switch (image.depth()) {
		case 1:
		{
			switch (image.channels()) {
				case 1: format = QImage::Format_Indexed8; break;
				case 3: format = QImage::Format_RGB888; break;
#ifdef WINDOWS
				case 4: format = QImage::Format_RGBA8888; break;
#endif
				default: std::cerr << "Unsupported image format.\n"; return;
			}
		}
		break;
		case 2:
		{
			switch (image.channels()) {
				case 3: format = QImage::Format_RGB16;
				default: std::cerr << "Unsupported image format.\n"; return;
			}
		}
		break;
		case 4:
		{
			switch (image.channels()) {
				case 3: format = QImage::Format_RGB32;
				default: std::cerr << "Unsupported image format.\n"; return;
			}
		}
		default:
			std::cerr << "Unsupported image format.\n";
			return;
	}

	// TODO: find neat way to show non-contiguous images.
	if (image.is_contiguous()) {
		this->image_lbl->setPixmap(
				QPixmap::fromImage(QImage(reinterpret_cast<byte*>(image.data()),
						image.cols(), image.rows(), format)));
	} else {
		auto im_c = image.clone();
		this->image_lbl->setPixmap(
				QPixmap::fromImage(QImage(reinterpret_cast<byte*>(im_c.data()),
						im_c.cols(), im_c.rows(), format)));
	}

	this->resize(image.cols(), image.rows());
}

void image_window::keyPressEvent(QKeyEvent *e) {
	global_image_application::singleton()->assign_key(e->text()[0].toAscii());
}

void image_window::closeEvent(QCloseEvent *e) {
}

/****************************************************************************
** Meta object code from reading C++ file 'gui.hpp'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.6)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'gui.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.6. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_cv__image_window[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

static const char qt_meta_stringdata_cv__image_window[] = {
    "cv::image_window\0"
};

void cv::image_window::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

const QMetaObjectExtraData cv::image_window::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject cv::image_window::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_cv__image_window,
      qt_meta_data_cv__image_window, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &cv::image_window::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *cv::image_window::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *cv::image_window::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_cv__image_window))
        return static_cast<void*>(const_cast< image_window*>(this));
    return QDialog::qt_metacast(_clname);
}

int cv::image_window::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    return _id;
}
QT_END_MOC_NAMESPACE

// MOC Generated code end /////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////
// class global_image_application

global_image_application* global_image_application::_singleton = nullptr;

global_image_application::global_image_application(int &argc, char** argv) : QApplication(argc, argv), _last_key('\0') {

}

global_image_application::~global_image_application() {
	this->destroy_all_windows();
}

global_image_application *global_image_application::singleton() {
	if (!global_image_application::_singleton) {
		int argc_simulation = 1;
		char *argv_simulation[] = {" ", " "};
		global_image_application::_singleton = new global_image_application(argc_simulation, argv_simulation);
	}
	return global_image_application::_singleton;
}

void global_image_application::force_quit() {
	this->quit();
}

void global_image_application::assign_key(char key) {
	this->_last_key = key;
	for (auto w : this->_windows) {
		w->close();
	}
}

char global_image_application::get_last_key() const {
	return this->_last_key;
}

void global_image_application::create_window(const std::string &name, const image_array &image) {
	image_window *newWindow = new image_window;
	newWindow->setWindowTitle(name.c_str());
	newWindow->set_image(image);
	newWindow->show();
	this->_windows.push_back(newWindow);
}

void global_image_application::destroy_window(const std::string &name) {
	int winId = -1;
	for (int i = 0; i < this->_windows.size(); i++) {
		if (this->_windows[i]->windowTitle() == name.c_str()) {
			winId = i;
			break;
		}
	}
	if (winId != -1) {
		this->_windows[winId]->close();
		delete this->_windows[winId];
		this->_windows.erase(this->_windows.begin() + winId);
	} else {
		std::cerr << ("Window by name " + name + " does not exist.\n");
	}
}

void global_image_application::destroy_all_windows() {
	for (auto window : this->_windows) {
		window->close();
		delete window;
	}
	this->_windows.clear();
}

void imshow(const std::string &name, const image_array &image) {
	ASSERT(image.is_valid() && (image.channels() == 1 || image.channels() == 3 || image.channels() == 4));
	if (image.depth() != 1) {
		auto byte_copy = image.clone();
		byte_copy.convert_to<byte>();
		global_image_application::singleton()->create_window(name, byte_copy);
	} else {
		global_image_application::singleton()->create_window(name, image);
	}
}

void imclose(const std::string &name) {
	global_image_application::singleton()->destroy_window(name);
}

void imclose_all() {
	global_image_application::singleton()->destroy_all_windows();
}

char wait_key() {
	auto ga_inst = global_image_application::singleton();
	ga_inst->exec();
	return ga_inst->get_last_key();
}

}

#endif // CV_IGNORE_GUI
