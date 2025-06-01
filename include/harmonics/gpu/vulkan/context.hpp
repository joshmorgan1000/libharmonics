#pragma once

#include "harmonics/config.hpp"
#include "harmonics/memory_profiler.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <optional>
#include <vector>

#if HARMONICS_HAS_VULKAN && __has_include(<vulkan/vulkan.h>)
#define HARMONICS_USE_VULKAN_RT 1
#include <vulkan/vulkan.h>
#else
#define HARMONICS_USE_VULKAN_RT 0
#endif

namespace harmonics {

inline std::optional<uint32_t>& vulkan_device_override() {
    static std::optional<uint32_t> index;
    return index;
}

inline void set_vulkan_device_index(uint32_t index) { vulkan_device_override() = index; }

#if HARMONICS_USE_VULKAN_RT

struct VulkanBuffer {
    VkDevice device{VK_NULL_HANDLE};
    VkBuffer buffer{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    std::size_t size{0};
};

struct VulkanContext {
    VkInstance instance{VK_NULL_HANDLE};
    VkPhysicalDevice physical{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    uint32_t device_index{0};
};

inline uint32_t find_host_visible_memory(uint32_t filter, VkPhysicalDevice physical) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((filter & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            return i;
        }
    }
    return 0;
}

inline uint32_t select_best_vulkan_device(const std::vector<VkPhysicalDevice>& devices) {
    uint32_t best = 0;
#if HARMONICS_USE_VULKAN_RT
    std::size_t best_score = 0;
    for (uint32_t i = 0; i < devices.size(); ++i) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        VkPhysicalDeviceMemoryProperties mem;
        vkGetPhysicalDeviceMemoryProperties(devices[i], &mem);
        std::size_t mem_size = 0;
        for (uint32_t h = 0; h < mem.memoryHeapCount; ++h)
            if (mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                mem_size += mem.memoryHeaps[h].size;
        std::size_t score = mem_size;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            score += mem_size; // favour discrete GPUs
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
#else
    (void)devices;
#endif
    return best;
}

inline VulkanContext& get_vulkan_context() {
    static VulkanContext ctx;
    static bool init = false;
    if (!init) {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "harmonics";
        app.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &app;
        vkCreateInstance(&ci, nullptr, &ctx.instance);

        uint32_t count = 0;
        vkEnumeratePhysicalDevices(ctx.instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(ctx.instance, &count, devices.data());

        uint32_t index = 0;
        if (const char* env = std::getenv("HARMONICS_VULKAN_DEVICE")) {
            int idx = std::atoi(env);
            if (idx >= 0 && static_cast<uint32_t>(idx) < count)
                index = static_cast<uint32_t>(idx);
        } else if (vulkan_device_override()) {
            uint32_t idx = *vulkan_device_override();
            if (idx < count)
                index = idx;
#if HARMONICS_USE_VULKAN_RT
        } else {
            index = select_best_vulkan_device(devices);
#endif
        }

        ctx.physical = devices[index];
        ctx.device_index = index;

        float priority = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = 0;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;

        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        vkCreateDevice(ctx.physical, &dci, nullptr, &ctx.device);

        init = true;
    }
    return ctx;
}

inline uint32_t vulkan_device_index() { return get_vulkan_context().device_index; }

struct VulkanPipeline {
    VkDevice device{VK_NULL_HANDLE};
    VkPipelineLayout layout{VK_NULL_HANDLE};
    VkPipeline pipeline{VK_NULL_HANDLE};
    VkShaderModule shader{VK_NULL_HANDLE};
};

inline VulkanPipeline create_compute_pipeline(const std::vector<uint32_t>& spirv) {
    VulkanPipeline pipe{};
#if HARMONICS_USE_VULKAN_RT
    auto& ctx = get_vulkan_context();
    pipe.device = ctx.device;
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spirv.size() * sizeof(uint32_t);
    smci.pCode = spirv.data();
    vkCreateShaderModule(ctx.device, &smci, nullptr, &pipe.shader);
    VkPipelineLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    vkCreatePipelineLayout(ctx.device, &lci, nullptr, &pipe.layout);
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = pipe.shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pci.stage = stage;
    pci.layout = pipe.layout;
    vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipe.pipeline);
#else
    (void)spirv;
#endif
    return pipe;
}

inline void destroy_pipeline(VulkanPipeline& pipe) {
#if HARMONICS_USE_VULKAN_RT
    if (pipe.pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(pipe.device, pipe.pipeline, nullptr);
    if (pipe.layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(pipe.device, pipe.layout, nullptr);
    if (pipe.shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(pipe.device, pipe.shader, nullptr);
#else
    (void)pipe;
#endif
    pipe.pipeline = VK_NULL_HANDLE;
    pipe.layout = VK_NULL_HANDLE;
    pipe.shader = VK_NULL_HANDLE;
    pipe.device = VK_NULL_HANDLE;
}

/**
 * @brief Dispatch a compute pipeline.
 *
 * Launches the given pipeline with the specified work group counts. On the
 * fallback path this function performs no work.
 */
inline void dispatch_compute_pipeline(const VulkanPipeline& pipe, uint32_t x, uint32_t y = 1,
                                      uint32_t z = 1) {
#if HARMONICS_USE_VULKAN_RT
    auto& ctx = get_vulkan_context();
    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = 0;
    VkCommandPool pool;
    vkCreateCommandPool(ctx.device, &cpci, nullptr, &pool);

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(ctx.device, &ai, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &bi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
    vkCmdDispatch(cmd, x, y, z);
    vkEndCommandBuffer(cmd);

    VkQueue queue;
    vkGetDeviceQueue(ctx.device, 0, 0, &queue);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkDestroyCommandPool(ctx.device, pool, nullptr);
#else
    (void)pipe;
    (void)x;
    (void)y;
    (void)z;
#endif
}

/**
 * @brief Dispatch a compute pipeline asynchronously.
 *
 * Launches the given pipeline on a background thread, returning a future
 * that resolves when execution finishes. On the fallback path the future
 * resolves immediately.
 */
inline std::future<void> dispatch_compute_pipeline_async(const VulkanPipeline& pipe, uint32_t x,
                                                         uint32_t y = 1, uint32_t z = 1) {
#if HARMONICS_USE_VULKAN_RT
    return std::async(std::launch::async, [=]() { dispatch_compute_pipeline(pipe, x, y, z); });
#else
    (void)pipe;
    (void)x;
    (void)y;
    (void)z;
    return std::async(std::launch::async, [] {});
#endif
}

inline VulkanBuffer vulkan_malloc(std::size_t bytes) {
    auto& ctx = get_vulkan_context();
    VulkanBuffer buf;
    buf.device = ctx.device;

    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(ctx.device, &bi, nullptr, &buf.buffer);

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx.device, buf.buffer, &req);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_host_visible_memory(req.memoryTypeBits, ctx.physical);
    vkAllocateMemory(ctx.device, &ai, nullptr, &buf.memory);

    vkBindBufferMemory(ctx.device, buf.buffer, buf.memory, 0);
    buf.size = bytes;
    return buf;
}

inline void vulkan_memcpy_to_device(VulkanBuffer& dst, const void* src, std::size_t bytes) {
    auto start = std::chrono::high_resolution_clock::now();
    void* map = nullptr;
    vkMapMemory(dst.device, dst.memory, 0, bytes, 0, &map);
    std::memcpy(map, src, bytes);
    vkUnmapMemory(dst.device, dst.memory);
    auto end = std::chrono::high_resolution_clock::now();
    record_memcpy_to_device(
        bytes, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    dst.size = bytes;
}

inline void vulkan_memcpy_to_host(void* dst, const VulkanBuffer& src, std::size_t bytes) {
    auto start = std::chrono::high_resolution_clock::now();
    void* map = nullptr;
    vkMapMemory(src.device, src.memory, 0, bytes, 0, &map);
    std::memcpy(dst, map, bytes);
    vkUnmapMemory(src.device, src.memory);
    auto end = std::chrono::high_resolution_clock::now();
    record_memcpy_to_host(
        bytes, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

inline std::future<void> vulkan_memcpy_to_device_async(VulkanBuffer& dst, const void* src,
                                                       std::size_t bytes) {
    return std::async(std::launch::async,
                      [&dst, src, bytes]() { vulkan_memcpy_to_device(dst, src, bytes); });
}

inline std::future<void> vulkan_memcpy_to_host_async(void* dst, const VulkanBuffer& src,
                                                     std::size_t bytes) {
    return std::async(std::launch::async,
                      [dst, &src, bytes]() { vulkan_memcpy_to_host(dst, src, bytes); });
}

inline void vulkan_free(VulkanBuffer& buf) {
    if (buf.buffer != VK_NULL_HANDLE)
        vkDestroyBuffer(buf.device, buf.buffer, nullptr);
    if (buf.memory != VK_NULL_HANDLE)
        vkFreeMemory(buf.device, buf.memory, nullptr);
    buf.buffer = VK_NULL_HANDLE;
    buf.memory = VK_NULL_HANDLE;
    buf.device = VK_NULL_HANDLE;
    buf.size = 0;
}

#else // HARMONICS_USE_VULKAN_RT

/** Simple host-backed buffer used to emulate Vulkan device memory. */
struct VulkanBuffer {
    std::vector<std::byte> data{};
};

inline VulkanBuffer vulkan_malloc(std::size_t bytes) {
    return VulkanBuffer{std::vector<std::byte>(bytes)};
}

inline void vulkan_memcpy_to_device(VulkanBuffer& dst, const void* src, std::size_t bytes) {
    auto start = std::chrono::high_resolution_clock::now();
    dst.data.resize(bytes);
    std::memcpy(dst.data.data(), src, bytes);
    auto end = std::chrono::high_resolution_clock::now();
    record_memcpy_to_device(
        bytes, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

inline void vulkan_memcpy_to_host(void* dst, const VulkanBuffer& src, std::size_t bytes) {
    auto start = std::chrono::high_resolution_clock::now();
    std::memcpy(dst, src.data.data(), bytes);
    auto end = std::chrono::high_resolution_clock::now();
    record_memcpy_to_host(
        bytes, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

inline std::future<void> vulkan_memcpy_to_device_async(VulkanBuffer& dst, const void* src,
                                                       std::size_t bytes) {
    return std::async(std::launch::async,
                      [&dst, src, bytes]() { vulkan_memcpy_to_device(dst, src, bytes); });
}

inline std::future<void> vulkan_memcpy_to_host_async(void* dst, const VulkanBuffer& src,
                                                     std::size_t bytes) {
    return std::async(std::launch::async,
                      [dst, &src, bytes]() { vulkan_memcpy_to_host(dst, src, bytes); });
}

inline uint32_t vulkan_device_index() {
    if (vulkan_device_override())
        return *vulkan_device_override();
    return 0;
}

#endif // HARMONICS_USE_VULKAN_RT

} // namespace harmonics
