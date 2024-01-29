#include <xgfx/window.h>
#include <xgfx/drawing.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#define MAX_BOUNCES 10
#define RAYS_PER_PIXEL 1

typedef float vec3[3];
typedef vec3 mat3[3];

vec3 pixbuf[640*480];

typedef struct {
    vec3 position;
    vec3 direction;
} Ray;

typedef struct {
    vec3 position;
    vec3 rotation;
    int fwd;
    int back;
    int left;
    int right;
    int rotLeft;
    int rotRight;
    int rotUp;
    int rotDown;
} Camera;
Camera camera;

typedef struct {
    vec3 position;
    vec3 color;
    vec3 emissionColor;
    vec3 specularColor;
    float radius;
    float emissionStrength;
    float smoothness;
    float specularProbability;
} Sphere;
Sphere spheres[3];
Sphere startSpheres[3];

void mul_vec3_mat3(vec3 vector, mat3 matrix, vec3 dest) {
    dest[0] = (matrix[0][0] * vector[0]) + (matrix[0][1] * vector[1]) + (matrix[0][2] * vector[2]);
    dest[1] = (matrix[1][0] * vector[0]) + (matrix[1][1] * vector[1]) + (matrix[1][2] * vector[2]);
    dest[2] = (matrix[2][0] * vector[0]) + (matrix[2][1] * vector[1]) + (matrix[2][2] * vector[2]);
}

void rotationMatrixXYZ(float angleX, float angleY, float angleZ, mat3 rotationMatrix) {
    rotationMatrix[0][0] = cos(angleY)*cos(angleZ); rotationMatrix[0][1] = sin(angleX)*sin(angleY)*cos(angleZ)-cos(angleX)*sin(angleZ); rotationMatrix[0][2] = cos(angleX)*sin(angleY)*cos(angleZ)+sin(angleX)*sin(angleZ);
    rotationMatrix[1][0] = cos(angleY)*sin(angleZ); rotationMatrix[1][1] = sin(angleX)*sin(angleY)*sin(angleZ)+cos(angleX)*cos(angleZ); rotationMatrix[1][2] = cos(angleX)*sin(angleY)*sin(angleZ)-sin(angleX)*cos(angleZ);
    rotationMatrix[2][0] = -sin(angleY); rotationMatrix[2][1] = sin(angleX)*cos(angleY); rotationMatrix[2][2] = cos(angleX)*cos(angleY);
}

void normalize(vec3 vector) {
    float length = sqrtf(vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2]);
    vector[0] /= length;
    vector[1] /= length;
    vector[2] /= length;
}

float distance(vec3 point1, vec3 point2) {
    return sqrtf((point1[0]-point2[0])*(point1[0]-point2[0]) + (point1[1]-point2[1])*(point1[1]-point2[1]) + (point1[2]-point2[2])*(point1[2]-point2[2]));
}

void lerp(vec3 a, vec3 b, vec3 result, float t) {
    result[0] = a[0] + (b[0]-a[0])*t;
    result[1] = a[1] + (b[1]-a[1])*t;
    result[2] = a[2] + (b[2]-a[2])*t;
}

void generateRandomPointOnHemisphere(vec3 point, vec3 normal) {
    // Generate random spherical coordinates
    float theta = ((float)rand() / RAND_MAX) * 2 * M_PI;  // Azimuthal angle
    float phi = ((float)rand() / RAND_MAX) * M_PI / 2;      // Polar angle (restricted to the upper hemisphere)

    // Convert spherical coordinates to Cartesian coordinates
    point[0] = sin(phi) * cos(theta);
    point[1] = sin(phi) * sin(theta);
    point[2] = cos(phi);

    // flip vector if dot product is negative
    float dotProduct = point[0] * normal[0] + point[1] * normal[1] + point[2] * normal[2];
    if (dotProduct < 0) {
        point[0] = -point[0];
        point[1] = -point[1];
        point[2] = -point[2];
    }
}

vec3 distanceComparator;
int compSphere(const void* a, const void* b) {
    float distanceToA = distance(distanceComparator, ((Sphere*)a)->position);
    float distanceToB = distance(distanceComparator, ((Sphere*)b)->position);
    if (distanceToA > distanceToB) return 1;
    if (distanceToA == distanceToB) return 0;
    if (distanceToA < distanceToB) return -1;
}

int doesIntersectSphere(vec3 vectorPos, vec3 vectorDir, vec3 sphere, float radius, vec3 intersectionPoint1, vec3 intersectionPoint2) {
    // Calculate the vector from the vectorPos to the sphere's center
    vec3 centerToPos;
    centerToPos[0] = vectorPos[0] - sphere[0];
    centerToPos[1] = vectorPos[1] - sphere[1];
    centerToPos[2] = vectorPos[2] - sphere[2];

    // Calculate the coefficients for the quadratic equation
    float a = vectorDir[0] * vectorDir[0] + vectorDir[1] * vectorDir[1] + vectorDir[2] * vectorDir[2];
    float b = 2 * (centerToPos[0] * vectorDir[0] + centerToPos[1] * vectorDir[1] + centerToPos[2] * vectorDir[2]);
    float c = centerToPos[0] * centerToPos[0] + centerToPos[1] * centerToPos[1] + centerToPos[2] * centerToPos[2] - radius * radius;

    // Calculate the dot product
    float dotProduct = centerToPos[0] * vectorDir[0] + centerToPos[1] * vectorDir[1] + centerToPos[2] * vectorDir[2];
    if (dotProduct < 0) {
        return 0;
    }

    // Calculate the discriminant for the quadratic equation
    float discriminant = b * b - 4 * a * c;

    // Check if the discriminant is non-negative to determine if there is an intersection
    if (discriminant >= 0) {
        // Calculate the two possible solutions for t (parameter along the ray)
        float t1 = (-b + sqrt(discriminant)) / (2 * a);
        float t2 = (-b - sqrt(discriminant)) / (2 * a);

        // Calculate the intersection points
        intersectionPoint1[0] = vectorPos[0] + t1 * vectorDir[0];
        intersectionPoint1[1] = vectorPos[1] + t1 * vectorDir[1];
        intersectionPoint1[2] = vectorPos[2] + t1 * vectorDir[2];

        intersectionPoint2[0] = vectorPos[0] + t2 * vectorDir[0];
        intersectionPoint2[1] = vectorPos[1] + t2 * vectorDir[1];
        intersectionPoint2[2] = vectorPos[2] + t2 * vectorDir[2];

        return 1;
    }

    return 0;
}

void trace(Ray ray, vec3 color) {
    vec3 rayColor;
    rayColor[0] = 1;
    rayColor[1] = 1;
    rayColor[2] = 1;
    vec3 incomingLight;
    incomingLight[0] = 0;
    incomingLight[1] = 0;
    incomingLight[2] = 0;

    for (int i = 0; i < MAX_BOUNCES; i++) {
        int hit = 0;
        qsort(spheres, 3, sizeof(Sphere), compSphere);
        for (int i = 0; i < 3; i++) {
            vec3 point1;
            vec3 point2;
            if (doesIntersectSphere(ray.position, ray.direction, spheres[i].position, spheres[i].radius, point1, point2)) {
                vec3 normal;
                normal[0] = spheres[i].position[0] - point1[0];
                normal[1] = spheres[i].position[1] - point1[1];
                normal[2] = spheres[i].position[2] - point1[2];
                normalize(normal);
                memcpy(ray.position, point1, sizeof(vec3));
                memcpy(distanceComparator, ray.position, sizeof(vec3));
                vec3 randomDirection;
                generateRandomPointOnHemisphere(randomDirection, normal);
                vec3 diffuseDirection;
                diffuseDirection[0] = normal[0] + randomDirection[0];
                diffuseDirection[1] = normal[1] + randomDirection[1];
                diffuseDirection[2] = normal[2] + randomDirection[2];
                normalize(diffuseDirection);
                vec3 specularDirection;
                float dotInNormal = ray.direction[0] * normal[0] + ray.direction[1] * normal[1] + ray.direction[2] * normal[2];
                specularDirection[0] = ray.direction[0] - 2*dotInNormal*normal[0];
                specularDirection[1] = ray.direction[1] - 2*dotInNormal*normal[1];
                specularDirection[2] = ray.direction[2] - 2*dotInNormal*normal[2];
                float isSpecular = spheres[i].specularProbability >= ((float)rand() / RAND_MAX);
                lerp(diffuseDirection, specularDirection, ray.direction, spheres[i].smoothness * isSpecular);
                vec3 emittedLight;
                emittedLight[0] = spheres[i].emissionColor[0] * spheres[i].emissionStrength;
                emittedLight[1] = spheres[i].emissionColor[1] * spheres[i].emissionStrength;
                emittedLight[2] = spheres[i].emissionColor[2] * spheres[i].emissionStrength;
                incomingLight[0] += emittedLight[0] * rayColor[0];
                incomingLight[1] += emittedLight[1] * rayColor[1];
                incomingLight[2] += emittedLight[2] * rayColor[2];
                vec3 newColor;
                lerp(spheres[i].color, spheres[i].specularColor, newColor, isSpecular);
                rayColor[0] *= newColor[0];
                rayColor[1] *= newColor[1];
                rayColor[2] *= newColor[2];
                hit = 1;
                break;
            }
        }
        if (!hit) {
            incomingLight[0] += 0.53 * 0.5 * rayColor[0];
            incomingLight[1] += 0.81 * 0.5 * rayColor[1];
            incomingLight[2] += 0.92 * 0.5 * rayColor[2];
            break;
        }
    }
    memcpy(color, incomingLight, sizeof(vec3));
}

XEvent eventBuffer[100];

int main() {
    initWindow(640, 480, "Raytracer");

    camera.position[0] = 7.329734;
    camera.position[1] = 0.000000;
    camera.position[2] = 2.851753;
    camera.rotation[0] = -0.100000; 
    camera.rotation[1] = 1.241594;
    camera.rotation[2] = 0.000000;
    float focalLength = 1;
    startSpheres[0].position[0] = 0;
    startSpheres[0].position[1] = 0;
    startSpheres[0].position[2] = 400;
    startSpheres[0].color[0] = 0;
    startSpheres[0].color[1] = 0;
    startSpheres[0].color[2] = 0;
    startSpheres[0].emissionColor[0] = 1;
    startSpheres[0].emissionColor[1] = 1;
    startSpheres[0].emissionColor[2] = 1;
    startSpheres[0].radius = 200;
    startSpheres[0].emissionStrength = 1;
    startSpheres[0].smoothness = 0;
    startSpheres[0].specularProbability = 0;
    startSpheres[1].position[0] = 0;
    startSpheres[1].position[1] = 0;
    startSpheres[1].position[2] = 3;
    startSpheres[1].specularColor[0] = 0.9;
    startSpheres[1].specularColor[1] = 0.9;
    startSpheres[1].specularColor[2] = 0.9;
    startSpheres[1].radius = 1;
    startSpheres[1].emissionStrength = 0;
    startSpheres[1].smoothness = 1;
    startSpheres[1].specularProbability = 1;
    startSpheres[2].position[0] = 0;
    startSpheres[2].position[1] = -2;
    startSpheres[2].position[2] = 0;
    startSpheres[2].color[0] = 0;
    startSpheres[2].color[1] = 1;
    startSpheres[2].color[2] = 0;
    startSpheres[2].radius = 2;
    startSpheres[2].emissionStrength = 0;
    startSpheres[2].smoothness = 0;
    startSpheres[2].specularProbability = 0;

    int frames = 0;
    int file = open("offscreen.data", O_WRONLY | O_CREAT, 0664);
    int videoFrames = 0;

    while(1) {
        int eventsRead = checkWindowEvents(eventBuffer, 100);
        for (int i = 0; i < eventsRead; i++) {
            XEvent event = eventBuffer[i];
            if (event.type == ClosedWindow) {
                close(file);
                return 0;
            }
            if (event.type == KeyPress) {
                if (event.xkey.keycode == 25) {
                    camera.fwd = 1;
                }
                if (event.xkey.keycode == 39) {
                    camera.back = 1;
                }
                if (event.xkey.keycode == 38) {
                    camera.left = 1;
                }
                if (event.xkey.keycode == 40) {
                    camera.right = 1;
                }
                if (event.xkey.keycode == 113) {
                    camera.rotLeft = 1;
                }
                if (event.xkey.keycode == 114) {
                    camera.rotRight = 1;
                }
                if (event.xkey.keycode == 111) {
                    camera.rotUp = 1;
                }
                if (event.xkey.keycode == 116) {
                    camera.rotDown = 1;
                }
                if (event.xkey.keycode == 54) {
                    printf("camera position: (%f, %f, %f) camera rotation: (%f, %f, %f) frame: %d\n", camera.position[0], camera.position[1], camera.position[2], camera.rotation[0], camera.rotation[1], camera.rotation[2], frames);
                }
            }
            if (event.type == KeyRelease) {
                if (event.xkey.keycode == 25) {
                    camera.fwd = 0;
                }
                if (event.xkey.keycode == 39) {
                    camera.back = 0;
                }
                if (event.xkey.keycode == 38) {
                    camera.left = 0;
                }
                if (event.xkey.keycode == 40) {
                    camera.right = 0;
                }
                if (event.xkey.keycode == 113) {
                    camera.rotLeft = 0;
                }
                if (event.xkey.keycode == 114) {
                    camera.rotRight = 0;
                }
                if (event.xkey.keycode == 111) {
                    camera.rotUp = 0;
                }
                if (event.xkey.keycode == 116) {
                    camera.rotDown = 0;
                }
            }
        }
        if (camera.fwd) {
            camera.position[0] -= 0.1 * sinf(camera.rotation[1]);
            camera.position[2] -= 0.1 * cosf(camera.rotation[1]);
        }
        if (camera.back) {
            
            camera.position[0] += 0.1 * sinf(camera.rotation[1]);
            camera.position[2] += 0.1 * cosf(camera.rotation[1]);
        }
        if (camera.left) {
            camera.position[0] += 0.1 * sinf(camera.rotation[1] + M_PI_2);
            camera.position[2] += 0.1 * cosf(camera.rotation[1] + M_PI_2);
        }
        if (camera.right) {
            camera.position[0] -= 0.1 * sinf(camera.rotation[1] + M_PI_2);
            camera.position[2] -= 0.1 * cosf(camera.rotation[1] + M_PI_2);
        }
        if (camera.rotLeft) {
            camera.rotation[1] -= 0.1;
        }
        if (camera.rotRight) {
            camera.rotation[1] += 0.1;
        }
        if (camera.rotUp) {
            camera.rotation[0] += 0.1;
        }
        if (camera.rotDown) {
            camera.rotation[0] -= 0.1;
        }
        mat3 rotationMatrix;
        rotationMatrixXYZ(camera.rotation[0], camera.rotation[1], camera.rotation[2], rotationMatrix);
        memcpy(spheres, startSpheres, sizeof(startSpheres));
        for (int y = 0; y < 480; y++) {
            for (int x = 0; x < 640; x++) {
                Ray ray;
                memcpy(ray.position, camera.position, sizeof(vec3));
                vec3 screen;
                screen[0] = (float)(x-320)/320;
                screen[1] = (float)(y-240)/320;
                screen[2] = focalLength;
                mul_vec3_mat3(screen, rotationMatrix, ray.direction);
                normalize(ray.direction);

                memcpy(distanceComparator, ray.position, sizeof(vec3));

                vec3 oldColor;
                memcpy(oldColor, pixbuf[y*640+x], sizeof(vec3));
                vec3 newColor;
                newColor[0] = 0;
                newColor[1] = 0;
                newColor[2] = 0;
                for (int i = 0; i < RAYS_PER_PIXEL; i++) {
                    vec3 color;
                    trace(ray, color);
                    newColor[0] += color[0];
                    newColor[1] += color[1];
                    newColor[2] += color[2];
                }
                newColor[0] /= RAYS_PER_PIXEL;
                newColor[1] /= RAYS_PER_PIXEL;
                newColor[2] /= RAYS_PER_PIXEL;

                float weight = 1.0 / (frames + 1);

                pixbuf[y*640+x][0] = newColor[0];
                pixbuf[y*640+x][1] = newColor[1];
                pixbuf[y*640+x][2] = newColor[2];

                pixbuf[y*640+x][0] = oldColor[0]*(1-weight) + newColor[0]*weight;
                pixbuf[y*640+x][1] = oldColor[1]*(1-weight) + newColor[1]*weight;
                pixbuf[y*640+x][2] = oldColor[2]*(1-weight) + newColor[2]*weight;

                plot(x, y, 0xff000000 + ((int)(pixbuf[y*640+x][0]*255)<<16) + ((int)(pixbuf[y*640+x][1]*255)<<8) + (int)(pixbuf[y*640+x][2]*255));
            }
        }

        if (frames == 100) {
            int framebuffer[640*480];
            for (int i = 0; i < 640*480; i++) {
                framebuffer[i] = 0xff000000 + ((int)(pixbuf[i][2]*255)<<16) + ((int)(pixbuf[i][1]*255)<<8) + (int)(pixbuf[i][0]*255);
            }
            write(file, framebuffer, sizeof(framebuffer));
            frames = 0;
            startSpheres[1].position[0] = 3 * sinf((float)videoFrames / 10);
            startSpheres[1].position[2] = 3 * cosf((float)videoFrames / 10);
            videoFrames++;
            printf("rendered %d video frames\n", videoFrames);
        }

        updateWindow();
        frames++;
    }
}