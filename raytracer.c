#include <xgfx/window.h>
#include <xgfx/drawing.h>
#define __USE_MISC
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#define MAX_BOUNCES 5
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
Camera camera = {
    .position = {3.605456, 0.000000, 4.752173},
    .rotation = {0.300000, -2.258405, 0.000000}
};

typedef struct {
    vec3 color;
    vec3 emissionColor;
    vec3 specularColor;
    float emissionStrength;
    float smoothness;
    float specularProbability;
} Material;
Material triangleMaterial = {
    .color = {0, 0, 0},
    .specularColor = {0, 0, 1},
    .emissionStrength = 0,
    .smoothness = 0.9,
    .specularProbability = 1
};

typedef struct {
    vec3 position;
    float radius;
    Material material;
} Sphere;
Sphere spheres[3];

typedef struct {
    int p1;
    int p2;
    int p3;
    Material material;
} Triangle;
Triangle triangles[] = {
    {2, 3, 1},
    {2, 3, 4},
    {6, 7, 5},
    {6, 7, 8},
    {2, 8, 4},
    {2, 8, 6},
    {5, 3, 1},
    {5, 3, 7},
    {1, 6, 2},
    {1, 6, 5},
    {3, 8, 4},
    {3, 8, 7}
};
vec3 originalVertices[] = {
    {10, 10, 10},
    {1, 1, 1},
    {-1, 1, 1},
    {1, -1, 1},
    {-1, -1, 1},
    {1, 1, -1},
    {-1, 1, -1},
    {1, -1, -1},
    {-1, -1, -1}
};
vec3 vertices[12];

int globalX;
int globalY;

float dotProduct(vec3 v1, vec3 v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

void crossProduct(vec3 v1, vec3 v2, vec3 result) {
    result[0] = v1[1] * v2[2] - v1[2] * v2[1];
    result[1] = v1[2] * v2[0] - v1[0] * v2[2];
    result[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

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
    float dot_product = dotProduct(point, normal);
    if (dot_product < 0) {
        point[0] = -point[0];
        point[1] = -point[1];
        point[2] = -point[2];
    }
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
    float dot_product = dotProduct(centerToPos, vectorDir);
    if (dot_product > 0) {
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

int intersectTriangleLine(vec3 origin, vec3 direction, Triangle* triangle, vec3 intersectionPoint) {

    // Calculate normal vector of the triangle
    vec3 edge1, edge2, normal;
    edge1[0] = vertices[triangle->p2][0] - vertices[triangle->p1][0];
    edge1[1] = vertices[triangle->p2][1] - vertices[triangle->p1][1];
    edge1[2] = vertices[triangle->p2][2] - vertices[triangle->p1][2];

    edge2[0] = vertices[triangle->p3][0] - vertices[triangle->p1][0];
    edge2[1] = vertices[triangle->p3][1] - vertices[triangle->p1][1];
    edge2[2] = vertices[triangle->p3][2] - vertices[triangle->p1][2];

    normal[0] = edge1[1] * edge2[2] - edge1[2] * edge2[1];
    normal[1] = edge1[2] * edge2[0] - edge1[0] * edge2[2];
    normal[2] = edge1[0] * edge2[1] - edge1[1] * edge2[0];

    // Check if the line and the plane of the triangle are parallel
    float dotProduct = normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2];
    if (dotProduct == 0) {
        return 0;  // No intersection (parallel)
    }

    // Calculate distance from line origin to the plane of the triangle
    float d = normal[0] * vertices[triangle->p1][0] + normal[1] * vertices[triangle->p1][1] + normal[2] * vertices[triangle->p1][2];

    // Calculate t parameter for the line equation L(t) = P0 + t * d, where P0 is the line origin
    float t = (d - (normal[0] * origin[0] + normal[1] * origin[1] + normal[2] * origin[2])) / dotProduct;

    // Check if the intersection point is within the triangle
    float u, v, w;
    vec3 intersection;
    intersection[0] = origin[0] + t * direction[0];
    intersection[1] = origin[1] + t * direction[1];
    intersection[2] = origin[2] + t * direction[2];

    // Barycentric coordinates
    u = ((edge2[1] * (intersection[0] - vertices[triangle->p1][0]) - edge2[0] * (intersection[1] - vertices[triangle->p1][1])) /
         (edge1[0] * edge2[1] - edge1[1] * edge2[0]));
    v = -((edge1[1] * (intersection[0] - vertices[triangle->p1][0]) - edge1[0] * (intersection[1] - vertices[triangle->p1][1])) /
          (edge1[0] * edge2[1] - edge1[1] * edge2[0]));
    w = 1 - u - v;

    if (t > 0.000001 && u >= 0 && v >= 0 && (u + v) <= 1 && w >= 0) {
        memcpy(intersectionPoint, intersection, sizeof(vec3));
        return 1;  // Intersection point is within the triangle
    } else {
        return 0;  // Intersection point is outside the triangle
    }
}

/*int intersectTriangleLine(vec3 origin, vec3 direction, Triangle* triangle, vec3 intersectionPoint) {
    // Calculate normal vector of the triangle
    vec3 edge1, edge2, normal;
    edge1[0] = vertices[triangle->p2][0] - vertices[triangle->p1][0];
    edge1[1] = vertices[triangle->p2][1] - vertices[triangle->p1][1];
    edge1[2] = vertices[triangle->p2][2] - vertices[triangle->p1][2];

    edge2[0] = vertices[triangle->p3][0] - vertices[triangle->p1][0];
    edge2[1] = vertices[triangle->p3][1] - vertices[triangle->p1][1];
    edge2[2] = vertices[triangle->p3][2] - vertices[triangle->p1][2];

    normal[0] = edge1[1] * edge2[2] - edge1[2] * edge2[1];
    normal[1] = edge1[2] * edge2[0] - edge1[0] * edge2[2];
    normal[2] = edge1[0] * edge2[1] - edge1[1] * edge2[0];

    // Check if the line and the plane of the triangle are parallel
    float dotProduct = normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2];
    if (dotProduct == 0) {
        return 0;  // No intersection (parallel)
    }

    // Calculate distance from line origin to the plane of the triangle
    float d = normal[0] * vertices[triangle->p1][0] + normal[1] * vertices[triangle->p1][1] + normal[2] * vertices[triangle->p1][2];

    // Calculate t parameter for the line equation L(t) = P0 + t * d, where P0 is the line origin
    float t = (d - (normal[0] * origin[0] + normal[1] * origin[1] + normal[2] * origin[2])) / dotProduct;

    // Check if the intersection point is within the triangle
    vec3 intersection;
    intersection[0] = origin[0] + t * direction[0];
    intersection[1] = origin[1] + t * direction[1];
    intersection[2] = origin[2] + t * direction[2];

    float u, v, w;
    if (fabs(normal[0]) < FLT_EPSILON) {
        // Triangle is aligned with YZ plane
        u = (intersection[1] - vertices[triangle->p1][1]) / (edge1[1] != 0 ? edge1[1] : edge2[1]);
        v = (intersection[2] - vertices[triangle->p1][2]) / (edge1[2] != 0 ? edge1[2] : edge2[2]);
        w = 1 - u - v;
    } else if (fabs(normal[1]) < FLT_EPSILON) {
        // Triangle is aligned with XZ plane
        u = (intersection[0] - vertices[triangle->p1][0]) / (edge1[0] != 0 ? edge1[0] : edge2[0]);
        v = (intersection[2] - vertices[triangle->p1][2]) / (edge1[2] != 0 ? edge1[2] : edge2[2]);
        w = 1 - u - v;
    } else {
        // General case: Calculate barycentric coordinates
        u = ((edge2[1] * (intersection[0] - vertices[triangle->p1][0]) - edge2[0] * (intersection[1] - vertices[triangle->p1][1])) /
             (edge1[0] * edge2[1] - edge1[1] * edge2[0]));
        v = -((edge1[1] * (intersection[0] - vertices[triangle->p1][0]) - edge1[0] * (intersection[1] - vertices[triangle->p1][1])) /
              (edge1[0] * edge2[1] - edge1[1] * edge2[0]));
        w = 1 - u - v;
    }

    if (t > 0.000001 && u >= 0 && v >= 0 && (u + v) <= 1 && w >= 0) {
        memcpy(intersectionPoint, intersection, sizeof(vec3));
        return 1;  // Intersection point is within the triangle
    } else {
        return 0;  // Intersection point is outside the triangle
    }
}*/


int didHit(vec3 position, vec3 direction, vec3 closestPoint, vec3 normal, Material* material) {

    closestPoint[0] = __FLT_MAX__;
    closestPoint[1] = __FLT_MAX__;
    closestPoint[2] = __FLT_MAX__;

    int hit = 0;

    for (int i = 0; i < sizeof(spheres)/sizeof(Sphere); i++) {
        vec3 point1;
        vec3 point2;
        if (doesIntersectSphere(position, direction, spheres[i].position, spheres[i].radius, point1, point2) && (distance(position, point1) < distance(position, closestPoint))) {
            hit = 1;
            memcpy(closestPoint, point2, sizeof(vec3));
            normal[0] = point2[0] - spheres[i].position[0];
            normal[1] = point2[1] - spheres[i].position[1];
            normal[2] = point2[2] - spheres[i].position[2];
            *material = spheres[i].material;
        }
    }

    for (int i = 0; i < sizeof(triangles)/sizeof(Triangle); i++) {
        //printf("checking triangle %d\n", i);
        vec3 point;
        if (intersectTriangleLine(position, direction, &triangles[i], point) && (distance(position, point) < distance(position, closestPoint))) {
            hit = 1;
            memcpy(closestPoint, point, sizeof(vec3));
            vec3 leg1;
            leg1[0] = vertices[triangles[i].p2][0] - vertices[triangles[i].p1][0];
            leg1[1] = vertices[triangles[i].p2][1] - vertices[triangles[i].p1][1];
            leg1[2] = vertices[triangles[i].p2][2] - vertices[triangles[i].p1][2];
            vec3 leg2;
            leg2[0] = vertices[triangles[i].p3][0] - vertices[triangles[i].p1][0];
            leg2[1] = vertices[triangles[i].p3][1] - vertices[triangles[i].p1][1];
            leg2[2] = vertices[triangles[i].p3][2] - vertices[triangles[i].p1][2];
            crossProduct(leg1, leg2, normal);
            *material = triangles[i].material;
        }
    }

    if (hit) {
        normalize(normal);
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
        vec3 point;
        vec3 normal;
        Material material;
        if (didHit(ray.position, ray.direction, point, normal, &material)) {
            memcpy(ray.position, point, sizeof(vec3));
            vec3 randomDirection;
            generateRandomPointOnHemisphere(randomDirection, normal);
            vec3 diffuseDirection;
            diffuseDirection[0] = normal[0] + randomDirection[0];
            diffuseDirection[1] = normal[1] + randomDirection[1];
            diffuseDirection[2] = normal[2] + randomDirection[2];
            normalize(diffuseDirection);
            vec3 specularDirection;
            float dotInNormal = dotProduct(ray.direction, normal);
            specularDirection[0] = ray.direction[0] - 2*dotInNormal*normal[0];
            specularDirection[1] = ray.direction[1] - 2*dotInNormal*normal[1];
            specularDirection[2] = ray.direction[2] - 2*dotInNormal*normal[2];
            float isSpecular = material.specularProbability >= ((float)rand() / RAND_MAX);
            lerp(diffuseDirection, specularDirection, ray.direction, material.smoothness * isSpecular);
            vec3 emittedLight;
            emittedLight[0] = material.emissionColor[0] * material.emissionStrength;
            emittedLight[1] = material.emissionColor[1] * material.emissionStrength;
            emittedLight[2] = material.emissionColor[2] * material.emissionStrength;
            incomingLight[0] += emittedLight[0] * rayColor[0];
            incomingLight[1] += emittedLight[1] * rayColor[1];
            incomingLight[2] += emittedLight[2] * rayColor[2];
            vec3 newColor;
            lerp(material.color, material.specularColor, newColor, isSpecular);
            rayColor[0] *= newColor[0];
            rayColor[1] *= newColor[1];
            rayColor[2] *= newColor[2];
        }
        else {
            incomingLight[0] += 0.53 * 0.5 * rayColor[0];
            incomingLight[1] += 0.81 * 0.5 * rayColor[1];
            incomingLight[2] += 0.92 * 0.5 * rayColor[2];
            break;
        }
    }
    memcpy(color, incomingLight, sizeof(vec3));
}

int main() {
    initWindow(640, 480, "Raytracer");
    float focalLength = 1;
    spheres[0].position[0] = 0;
    spheres[0].position[1] = 0;
    spheres[0].position[2] = 400;
    spheres[0].material.color[0] = 0;
    spheres[0].material.color[1] = 0;
    spheres[0].material.color[2] = 0;
    spheres[0].material.emissionColor[0] = 1;
    spheres[0].material.emissionColor[1] = 1;
    spheres[0].material.emissionColor[2] = 1;
    spheres[0].radius = 200;
    spheres[0].material.emissionStrength = 1;
    spheres[0].material.smoothness = 0;
    spheres[0].material.specularProbability = 0;
    spheres[1].position[0] = 0;
    spheres[1].position[1] = 0;
    spheres[1].position[2] = 3;
    spheres[1].material.color[0] = 1;
    spheres[1].material.color[1] = 0;
    spheres[1].material.color[2] = 0;
    spheres[1].material.specularColor[0] = 1;
    spheres[1].material.specularColor[1] = 1;
    spheres[1].material.specularColor[2] = 1;
    spheres[1].radius = 1;
    spheres[1].material.emissionStrength = 0;
    spheres[1].material.smoothness = 1;
    spheres[1].material.specularProbability = 0.2;
    spheres[2].position[0] = 0;
    spheres[2].position[1] = -4;
    spheres[2].position[2] = 0;
    spheres[2].material.color[0] = 0;
    spheres[2].material.color[1] = 1;
    spheres[2].material.color[2] = 0;
    spheres[2].radius = 2;
    spheres[2].material.emissionStrength = 0;
    spheres[2].material.smoothness = 0;
    spheres[2].material.specularProbability = 0;

    for (int i = 0; i < sizeof(triangles)/sizeof(Triangle); i++) {
        triangles[i].material = triangleMaterial;
        printf("Triangle (%f, %f, %f) (%f, %f, %f) (%f, %f, %f)\n", originalVertices[triangles[i].p1][0], originalVertices[triangles[i].p1][1], originalVertices[triangles[i].p1][2], originalVertices[triangles[i].p2][0], originalVertices[triangles[i].p2][1], originalVertices[triangles[i].p2][2], originalVertices[triangles[i].p3][0], originalVertices[triangles[i].p3][1], originalVertices[triangles[i].p3][2]);
    }

    int frames = 0;
    int file = open("offscreen.data", O_WRONLY | O_CREAT, 0664);
    int videoFrames = 0;

    while(1) {
        Event event;
        while (checkWindowEvent(&event)) {
            if (event.type == WINDOW_CLOSE) {
                close(file);
                return 0;
            }
            if (event.type == KEY_CHANGE) {
                if (event.keychange.key == 17) {
                    camera.fwd = event.keychange.state;
                }
                if (event.keychange.key == 31) {
                    camera.back = event.keychange.state;
                }
                if (event.keychange.key == 30) {
                    camera.left = event.keychange.state;
                }
                if (event.keychange.key == 32) {
                    camera.right = event.keychange.state;
                }
                if (event.keychange.key == 105) {
                    camera.rotLeft = event.keychange.state;
                }
                if (event.keychange.key == 106) {
                    camera.rotRight = event.keychange.state;
                }
                if (event.keychange.key == 103) {
                    camera.rotUp = event.keychange.state;
                }
                if (event.keychange.key == 108) {
                    camera.rotDown = event.keychange.state;
                }
                if (event.keychange.key == 46 && event.keychange.state == 1) {
                    printf("camera position: (%f, %f, %f) camera rotation: (%f, %f, %f) frame: %d\n", camera.position[0], camera.position[1], camera.position[2], camera.rotation[0], camera.rotation[1], camera.rotation[2], frames);
                }
            }
        }
        if (camera.fwd) {
            camera.position[0] += 0.1 * sinf(camera.rotation[1]);
            camera.position[2] += 0.1 * cosf(camera.rotation[1]);
        }
        if (camera.back) {
            
            camera.position[0] -= 0.1 * sinf(camera.rotation[1]);
            camera.position[2] -= 0.1 * cosf(camera.rotation[1]);
        }
        if (camera.left) {
            camera.position[0] -= 0.1 * sinf(camera.rotation[1] + M_PI_2);
            camera.position[2] -= 0.1 * cosf(camera.rotation[1] + M_PI_2);
        }
        if (camera.right) {
            camera.position[0] += 0.1 * sinf(camera.rotation[1] + M_PI_2);
            camera.position[2] += 0.1 * cosf(camera.rotation[1] + M_PI_2);
        }
        if (camera.rotLeft) {
            camera.rotation[1] -= 0.1;
        }
        if (camera.rotRight) {
            camera.rotation[1] += 0.1;
        }
        if (camera.rotUp) {
            camera.rotation[0] -= 0.1;
        }
        if (camera.rotDown) {
            camera.rotation[0] += 0.1;
        }
        mat3 rotationMatrix;
        rotationMatrixXYZ(camera.rotation[0], camera.rotation[1], camera.rotation[2], rotationMatrix);

        mat3 cubeMatrix;
        rotationMatrixXYZ((float)frames/5, (float)frames/5, 0, cubeMatrix);
        for (int i = 0; i < sizeof(vertices)/sizeof(vec3); i++) {
            mul_vec3_mat3(originalVertices[i], cubeMatrix, vertices[i]);
        }

        srand(frames*(videoFrames+1));
        for (int y = 0; y < 480; y++) {
            for (int x = 0; x < 640; x++) {
                globalX = x;
                globalY = y;
                Ray ray;
                memcpy(ray.position, camera.position, sizeof(vec3));
                vec3 screen;
                screen[0] = (float)(x-320)/320;
                screen[1] = (float)(y-240)/-320;
                screen[2] = focalLength;
                mul_vec3_mat3(screen, rotationMatrix, ray.direction);
                normalize(ray.direction);

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

                //pixbuf[y*640+x][0] = oldColor[0]*(1-weight) + newColor[0]*weight;
                //pixbuf[y*640+x][1] = oldColor[1]*(1-weight) + newColor[1]*weight;
                //pixbuf[y*640+x][2] = oldColor[2]*(1-weight) + newColor[2]*weight;

                plot(x, y, 0xff000000 + ((int)(pixbuf[y*640+x][0]*255)<<16) + ((int)(pixbuf[y*640+x][1]*255)<<8) + (int)(pixbuf[y*640+x][2]*255));
            }
        }
        /*if (frames == 50) {
            int framebuffer[640*480];
            for (int i = 0; i < 640*480; i++) {
                framebuffer[i] = 0xff000000 + ((int)(pixbuf[i][2]*255)<<16) + ((int)(pixbuf[i][1]*255)<<8) + (int)(pixbuf[i][0]*255);
            }
            write(file, framebuffer, sizeof(framebuffer));
            frames = 0;
            spheres[1].position[0] = 3 * sinf((float)videoFrames / 10);
            spheres[1].position[2] = 3 * cosf((float)videoFrames / 10);
            videoFrames++;
            printf("rendered %d video frames\n", videoFrames);
            
        }*/
        updateWindow();
        frames++;
    }
}